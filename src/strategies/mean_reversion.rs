//! Mean Reversion Trading Strategy
//!
//! This strategy identifies when prices deviate significantly from their mean
//! and trades expecting a reversion to the mean.

use crate::core::types::{InstrumentId, Price, Quantity};
use crate::market_data::events::MarketEvent;
use crate::strategy::output::{OrderRequest, StrategyMetrics};
use crate::strategy::{
    OrderSide, Strategy, StrategyConfig, StrategyContext, StrategyError, StrategyOutput,
};

/// Configuration for mean reversion strategy
#[derive(Debug, Clone)]
pub struct MeanReversionConfig {
    /// Lookback period for calculating mean
    pub lookback_period: usize,
    /// Standard deviations for entry signal
    pub entry_threshold: f64,
    /// Standard deviations for exit signal
    pub exit_threshold: f64,
    /// Maximum position size
    pub max_position_size: i64,
    /// Order size per trade
    pub order_size: u32,
    /// Use limit orders
    pub use_limit_orders: bool,
    /// Tick offset for limit orders
    pub limit_order_offset_ticks: i64,
}

impl Default for MeanReversionConfig {
    fn default() -> Self {
        Self {
            lookback_period: 20,
            entry_threshold: 2.0,
            exit_threshold: 0.5,
            max_position_size: 10,
            order_size: 1,
            use_limit_orders: true,
            limit_order_offset_ticks: 1,
        }
    }
}

/// Mean reversion trading strategy
pub struct MeanReversionStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// Mean reversion specific config
    mr_config: MeanReversionConfig,
    /// Price history for each instrument
    price_history: Vec<(u64, Price)>,
    /// Current mean
    current_mean: Option<f64>,
    /// Current standard deviation
    current_std: Option<f64>,
    /// Last signal generated
    last_signal: Signal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Signal {
    None,
    Long,
    Short,
    ExitLong,
    ExitShort,
}

impl MeanReversionStrategy {
    /// Create a new mean reversion strategy
    pub fn new(
        strategy_id: String,
        instrument_id: InstrumentId,
        mr_config: MeanReversionConfig,
    ) -> Self {
        let config = StrategyConfig::new(strategy_id, "Mean Reversion Strategy")
            .with_instrument(instrument_id)
            .with_max_position(mr_config.max_position_size);

        Self {
            config,
            mr_config: mr_config.clone(),
            price_history: Vec::with_capacity(mr_config.lookback_period * 2),
            current_mean: None,
            current_std: None,
            last_signal: Signal::None,
        }
    }

    /// Update price history and calculate statistics
    fn update_statistics(&mut self, price: Price, timestamp: u64) {
        // Add to price history
        self.price_history.push((timestamp, price));

        // Keep only required history
        if self.price_history.len() > self.mr_config.lookback_period * 2 {
            self.price_history.remove(0);
        }

        // Calculate mean and standard deviation if we have enough data
        if self.price_history.len() >= self.mr_config.lookback_period {
            let recent_prices: Vec<f64> = self
                .price_history
                .iter()
                .rev()
                .take(self.mr_config.lookback_period)
                .map(|(_, p)| p.as_f64())
                .collect();

            let mean = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;

            let variance = recent_prices
                .iter()
                .map(|p| (p - mean).powi(2))
                .sum::<f64>()
                / recent_prices.len() as f64;

            let std_dev = variance.sqrt();

            self.current_mean = Some(mean);
            self.current_std = Some(std_dev);
        }
    }

    /// Generate trading signal based on current price and statistics
    fn generate_signal(&mut self, current_price: f64, current_position: i64) -> Signal {
        if let (Some(mean), Some(std)) = (self.current_mean, self.current_std) {
            if std > 0.0 {
                let z_score = (current_price - mean) / std;

                // Entry signals
                if current_position == 0 {
                    if z_score < -self.mr_config.entry_threshold {
                        return Signal::Long; // Price too low, buy
                    } else if z_score > self.mr_config.entry_threshold {
                        return Signal::Short; // Price too high, sell
                    }
                }

                // Exit signals
                if current_position > 0 && z_score > -self.mr_config.exit_threshold {
                    return Signal::ExitLong; // Price reverted, exit long
                } else if current_position < 0 && z_score < self.mr_config.exit_threshold {
                    return Signal::ExitShort; // Price reverted, exit short
                }
            }
        }

        Signal::None
    }

    /// Create order request based on signal
    fn create_order(
        &self,
        signal: Signal,
        current_price: Price,
        context: &StrategyContext,
    ) -> Option<OrderRequest> {
        let instrument_id = self.config.instruments[0]; // Single instrument strategy

        let (side, quantity) = match signal {
            Signal::Long => (OrderSide::Buy, self.mr_config.order_size),
            Signal::Short => (OrderSide::SellShort, self.mr_config.order_size),
            Signal::ExitLong => {
                let pos = context.position.quantity.abs() as u32;
                (OrderSide::Sell, pos.min(self.mr_config.order_size))
            }
            Signal::ExitShort => {
                let pos = context.position.quantity.abs() as u32;
                (OrderSide::BuyCover, pos.min(self.mr_config.order_size))
            }
            Signal::None => return None,
        };

        if quantity == 0 {
            return None;
        }

        if self.mr_config.use_limit_orders {
            // Place limit order with offset
            let tick_size = 25; // TODO: Get from instrument config
            let offset = self.mr_config.limit_order_offset_ticks * tick_size;

            let limit_price = match side {
                OrderSide::Buy | OrderSide::BuyCover => Price::new(current_price.0 - offset),
                OrderSide::Sell | OrderSide::SellShort => Price::new(current_price.0 + offset),
            };

            Some(OrderRequest::limit_order(
                context.strategy_id.clone(),
                instrument_id,
                side,
                limit_price,
                Quantity::from(quantity),
            ))
        } else {
            Some(OrderRequest::market_order(
                context.strategy_id.clone(),
                instrument_id,
                side,
                Quantity::from(quantity),
            ))
        }
    }
}

impl Strategy for MeanReversionStrategy {
    fn initialize(&mut self, _context: &StrategyContext) -> Result<(), StrategyError> {
        self.price_history.clear();
        self.current_mean = None;
        self.current_std = None;
        self.last_signal = Signal::None;
        Ok(())
    }

    fn on_market_event(
        &mut self,
        event: &MarketEvent,
        context: &StrategyContext,
    ) -> StrategyOutput {
        let mut output = StrategyOutput::default();

        // Extract price from market event
        let price = match event {
            MarketEvent::Trade(trade) => {
                if trade.instrument_id == self.config.instruments[0] {
                    Some(trade.price)
                } else {
                    None
                }
            }
            MarketEvent::BBO(bbo) => {
                if bbo.instrument_id == self.config.instruments[0] {
                    // Use mid price
                    match (bbo.bid_price, bbo.ask_price) {
                        (Some(bid), Some(ask)) => {
                            Some(Price::from_f64((bid.as_f64() + ask.as_f64()) / 2.0))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(price) = price {
            // Update statistics
            self.update_statistics(price, event.timestamp());

            // Generate signal
            let signal = self.generate_signal(price.as_f64(), context.position.quantity);

            // Only act on signal changes to avoid duplicate orders
            if signal != self.last_signal && signal != Signal::None {
                if let Some(order) = self.create_order(signal, price, context) {
                    output.orders.push(order);
                    self.last_signal = signal;
                }
            }

            // Update metrics
            if let (Some(mean), Some(std)) = (self.current_mean, self.current_std) {
                let z_score = (price.as_f64() - mean) / std;
                let mut metrics = StrategyMetrics::new(event.timestamp());
                metrics.add("mean", mean);
                metrics.add("std_dev", std);
                metrics.add("z_score", z_score);
                metrics.add("signal", signal as i32 as f64);
                output.set_metrics(metrics);
            }
        }

        output
    }

    fn config(&self) -> &StrategyConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{FeaturePosition, RiskLimits};
    use crate::strategy::output::OrderType;

    #[test]
    fn test_mean_reversion_creation() {
        let strategy =
            MeanReversionStrategy::new("test_mr".to_string(), 1, MeanReversionConfig::default());

        assert_eq!(strategy.config.id, "test_mr");
        assert_eq!(strategy.config.instruments.len(), 1);
        assert_eq!(strategy.config.instruments[0], 1);
    }

    #[test]
    fn test_signal_generation() {
        let mut strategy = MeanReversionStrategy::new(
            "test_mr".to_string(),
            1,
            MeanReversionConfig {
                lookback_period: 5,
                entry_threshold: 2.0,
                exit_threshold: 0.5,
                ..Default::default()
            },
        );

        // Add some price history with known pattern that creates sufficient deviation
        // Mean will be (100+101+99+100+100)/5 = 100.0
        // Std will be sqrt(((0^2 + 1^2 + (-1)^2 + 0^2 + 0^2)/5)) = sqrt(2/5) = 0.632
        let prices = vec![100.0, 101.0, 99.0, 100.0, 100.0];

        for (i, price) in prices.iter().enumerate() {
            strategy.update_statistics(Price::from_f64(*price), i as u64 * 1000);
        }

        // Verify we have statistics
        assert!(strategy.current_mean.is_some(), "Mean should be calculated");
        assert!(strategy.current_std.is_some(), "Std should be calculated");

        let mean = strategy.current_mean.unwrap();
        let std = strategy.current_std.unwrap();

        println!("Mean: {}, Std: {}", mean, std);

        // Test with a price that should generate a long signal
        // Need z-score < -2.0, so price < mean - 2*std
        let low_price = mean - 2.1 * std; // Ensure z-score < -2.0
        let signal = strategy.generate_signal(low_price, 0);
        assert_eq!(
            signal,
            Signal::Long,
            "Low price {} should generate Long signal (z-score: {})",
            low_price,
            (low_price - mean) / std
        );

        // Test with a price that should generate a short signal
        let high_price = mean + 2.1 * std; // Ensure z-score > 2.0
        let signal = strategy.generate_signal(high_price, 0);
        assert_eq!(
            signal,
            Signal::Short,
            "High price {} should generate Short signal",
            high_price
        );

        // Test exit signal when in long position and price reverts
        let revert_price = mean - 0.4 * std; // z-score > -0.5
        let signal = strategy.generate_signal(revert_price, 1); // position = 1 (long)
        assert_eq!(
            signal,
            Signal::ExitLong,
            "Reverted price should generate ExitLong signal"
        );

        // Test exit signal when in short position and price reverts
        let revert_price = mean + 0.4 * std; // z-score < 0.5
        let signal = strategy.generate_signal(revert_price, -1); // position = -1 (short)
        assert_eq!(
            signal,
            Signal::ExitShort,
            "Reverted price should generate ExitShort signal"
        );
    }

    #[test]
    fn test_order_creation() {
        let strategy = MeanReversionStrategy::new(
            "test_mr".to_string(),
            1,
            MeanReversionConfig {
                order_size: 5,
                use_limit_orders: true,
                limit_order_offset_ticks: 2,
                ..Default::default()
            },
        );

        let context = StrategyContext::new(
            "test_mr".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        let order = strategy.create_order(Signal::Long, Price::new(100i64), &context);
        assert!(order.is_some());

        let order = order.unwrap();
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.quantity, Quantity::from(5u32));
        assert_eq!(order.order_type, OrderType::Limit);

        // Check limit price offset (2 ticks * 25 = 50)
        assert_eq!(order.price, Some(Price::new(100i64 - 50)));
    }

    #[test]
    fn test_config_validation() {
        // Test default configuration
        let default_config = MeanReversionConfig::default();
        assert_eq!(default_config.lookback_period, 20);
        assert_eq!(default_config.entry_threshold, 2.0);
        assert_eq!(default_config.exit_threshold, 0.5);
        assert_eq!(default_config.max_position_size, 10);
        assert_eq!(default_config.order_size, 1);
        assert_eq!(default_config.use_limit_orders, true);
        assert_eq!(default_config.limit_order_offset_ticks, 1);

        // Test custom configuration
        let custom_config = MeanReversionConfig {
            lookback_period: 30,
            entry_threshold: 2.5,
            exit_threshold: 0.75,
            max_position_size: 15,
            order_size: 3,
            use_limit_orders: false,
            limit_order_offset_ticks: 2,
        };

        let strategy =
            MeanReversionStrategy::new("custom_mr".to_string(), 2, custom_config.clone());

        assert_eq!(strategy.mr_config.lookback_period, 30);
        assert_eq!(strategy.mr_config.entry_threshold, 2.5);
        assert_eq!(strategy.mr_config.use_limit_orders, false);
    }

    #[test]
    fn test_strategy_initialization() {
        let mut strategy =
            MeanReversionStrategy::new("init_test".to_string(), 1, MeanReversionConfig::default());

        let context = StrategyContext::new(
            "init_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        // Add some data first
        strategy.price_history.push((500, Price::from_f64(100.0)));
        strategy.current_mean = Some(105.0);
        strategy.last_signal = Signal::Long;

        // Initialize strategy
        let result = strategy.initialize(&context);
        assert!(result.is_ok());

        // Check that state is properly reset
        assert!(strategy.price_history.is_empty());
        assert!(strategy.current_mean.is_none());
        assert!(strategy.current_std.is_none());
        assert_eq!(strategy.last_signal, Signal::None);
    }

    #[test]
    fn test_statistics_calculation() {
        let mut strategy = MeanReversionStrategy::new(
            "stats_test".to_string(),
            1,
            MeanReversionConfig {
                lookback_period: 5,
                ..Default::default()
            },
        );

        // Test with insufficient data
        strategy.update_statistics(Price::from_f64(100.0), 1000);
        strategy.update_statistics(Price::from_f64(102.0), 2000);
        assert!(strategy.current_mean.is_none());
        assert!(strategy.current_std.is_none());

        // Add enough data for statistics
        let prices = vec![100.0, 102.0, 98.0, 104.0, 96.0]; // Mean = 100.0
        for (i, price) in prices.iter().enumerate() {
            strategy.update_statistics(Price::from_f64(*price), (i as u64 + 1) * 1000);
        }

        assert!(strategy.current_mean.is_some());
        assert!(strategy.current_std.is_some());

        let mean = strategy.current_mean.unwrap();
        let std = strategy.current_std.unwrap();

        // Mean should be 100.0
        assert!(
            (mean - 100.0).abs() < 0.1,
            "Mean {} should be close to 100",
            mean
        );

        // Standard deviation should be > 0
        assert!(std > 0.0, "Standard deviation {} should be positive", std);

        // Test with more data to verify history management
        for i in 0..50 {
            let price = 100.0 + (i % 5) as f64 - 2.0; // Varies between 98-102
            strategy.update_statistics(Price::from_f64(price), (i as u64 + 10) * 1000);
        }

        // Should still have statistics
        assert!(strategy.current_mean.is_some());
        assert!(strategy.current_std.is_some());

        // Price history should be limited
        assert!(strategy.price_history.len() <= strategy.mr_config.lookback_period * 2);
    }

    #[test]
    fn test_comprehensive_signal_generation() {
        let mut strategy = MeanReversionStrategy::new(
            "signal_test".to_string(),
            1,
            MeanReversionConfig {
                lookback_period: 5,
                entry_threshold: 2.0,
                exit_threshold: 0.5,
                ..Default::default()
            },
        );

        // Set up known statistics
        // Mean = 100, Std = 2.0 (for easy calculation)
        strategy.current_mean = Some(100.0);
        strategy.current_std = Some(2.0);

        // Test no signal when price is near mean
        let signal = strategy.generate_signal(100.0, 0); // Z-score = 0
        assert_eq!(signal, Signal::None);

        // Test long entry signal (price too low)
        let signal = strategy.generate_signal(95.5, 0); // Z-score = (95.5-100)/2 = -2.25 < -2.0
        assert_eq!(signal, Signal::Long);

        // Test short entry signal (price too high)
        let signal = strategy.generate_signal(104.5, 0); // Z-score = (104.5-100)/2 = 2.25 > 2.0
        assert_eq!(signal, Signal::Short);

        // Test long exit signal (price reverted while long)
        let signal = strategy.generate_signal(99.2, 5); // Z-score = (99.2-100)/2 = -0.4 > -0.5
        assert_eq!(signal, Signal::ExitLong);

        // Test short exit signal (price reverted while short)
        let signal = strategy.generate_signal(100.8, -3); // Z-score = (100.8-100)/2 = 0.4 < 0.5
        assert_eq!(signal, Signal::ExitShort);

        // Test no exit when still far from mean
        let signal = strategy.generate_signal(97.0, 5); // Z-score = -1.5, still far
        assert_eq!(signal, Signal::None);

        // Test no entry when already in position
        let signal = strategy.generate_signal(95.0, 2); // Would be long signal but already long
        assert_eq!(signal, Signal::None);

        // Test with zero standard deviation
        strategy.current_std = Some(0.0);
        let signal = strategy.generate_signal(95.0, 0);
        assert_eq!(signal, Signal::None); // Should handle gracefully
    }

    #[test]
    fn test_order_creation_variants() {
        let strategy = MeanReversionStrategy::new(
            "order_test".to_string(),
            1,
            MeanReversionConfig {
                order_size: 5,
                use_limit_orders: true,
                limit_order_offset_ticks: 3,
                ..Default::default()
            },
        );

        let context = StrategyContext::new(
            "order_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        // Test long entry order (limit)
        let order = strategy.create_order(Signal::Long, Price::new(10000i64), &context);
        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.quantity, Quantity::from(5u32));
        assert_eq!(order.order_type, OrderType::Limit);
        // Price should be offset by 3 ticks * 25 = 75 lower
        assert_eq!(order.price, Some(Price::new(10000i64 - 75)));

        // Test short entry order (limit)
        let order = strategy.create_order(Signal::Short, Price::new(10000i64), &context);
        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.side, OrderSide::SellShort);
        // Price should be offset by 3 ticks * 25 = 75 higher
        assert_eq!(order.price, Some(Price::new(10000i64 + 75)));

        // Test exit long order
        let context_with_position = StrategyContext::new(
            "order_test".to_string(),
            1000,
            FeaturePosition {
                quantity: 3,
                ..Default::default()
            },
            RiskLimits::default(),
            true,
        );

        let order = strategy.create_order(
            Signal::ExitLong,
            Price::new(10000i64),
            &context_with_position,
        );
        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.side, OrderSide::Sell);
        assert_eq!(order.quantity, Quantity::from(3u32)); // Position size, not order size

        // Test exit short order
        let context_short = StrategyContext::new(
            "order_test".to_string(),
            1000,
            FeaturePosition {
                quantity: -4,
                ..Default::default()
            },
            RiskLimits::default(),
            true,
        );

        let order = strategy.create_order(Signal::ExitShort, Price::new(10000i64), &context_short);
        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.side, OrderSide::BuyCover);
        assert_eq!(order.quantity, Quantity::from(4u32));

        // Test no signal
        let order = strategy.create_order(Signal::None, Price::new(10000i64), &context);
        assert!(order.is_none());
    }

    #[test]
    fn test_market_orders() {
        let strategy = MeanReversionStrategy::new(
            "market_test".to_string(),
            1,
            MeanReversionConfig {
                order_size: 2,
                use_limit_orders: false, // Use market orders
                ..Default::default()
            },
        );

        let context = StrategyContext::new(
            "market_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        // Test market order creation
        let order = strategy.create_order(Signal::Long, Price::new(10000i64), &context);
        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.price, None); // Market orders don't have price
    }

    #[test]
    fn test_market_event_processing() {
        use crate::market_data::events::{BBOUpdate, MarketEvent, TradeEvent};

        let mut strategy = MeanReversionStrategy::new(
            "event_test".to_string(),
            1,
            MeanReversionConfig {
                lookback_period: 3,
                entry_threshold: 2.0,
                exit_threshold: 0.5,
                ..Default::default()
            },
        );

        let context = StrategyContext::new(
            "event_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        strategy.initialize(&context).unwrap();

        // Test trade event processing
        let trade_event = MarketEvent::Trade(TradeEvent {
            instrument_id: 1,
            trade_id: 12345,
            price: Price::from_f64(100.0),
            quantity: Quantity::from(10u32),
            aggressor_side: crate::core::types::Side::Bid,
            timestamp: 2000,
            buyer_order_id: None,
            seller_order_id: None,
        });

        let output = strategy.on_market_event(&trade_event, &context);
        assert!(output.orders.is_empty()); // No signal without enough data

        // Test BBO event processing
        let bbo_event = MarketEvent::BBO(BBOUpdate {
            instrument_id: 1,
            bid_price: Some(Price::from_f64(99.0)),
            ask_price: Some(Price::from_f64(101.0)),
            bid_quantity: Some(Quantity::from(5u32)),
            ask_quantity: Some(Quantity::from(8u32)),
            bid_order_count: Some(2),
            ask_order_count: Some(3),
            timestamp: 3000,
        });

        let output = strategy.on_market_event(&bbo_event, &context);
        assert!(output.orders.is_empty()); // Still not enough data

        // Add enough data to potentially generate signals
        for i in 0..5 {
            let price = 100.0 + (i % 3) as f64 - 1.0; // 99, 100, 101, 99, 100
            let trade = MarketEvent::Trade(TradeEvent {
                instrument_id: 1,
                trade_id: (12346 + i) as u64,
                price: Price::from_f64(price),
                quantity: Quantity::from(5u32),
                aggressor_side: crate::core::types::Side::Bid,
                timestamp: (4000 + i * 1000) as u64,
                buyer_order_id: None,
                seller_order_id: None,
            });

            let _output = strategy.on_market_event(&trade, &context);
            // Might generate orders once we have enough data
        }

        // Test event for different instrument (should be ignored)
        let other_trade = MarketEvent::Trade(TradeEvent {
            instrument_id: 2, // Different instrument
            trade_id: 67890,
            price: Price::from_f64(200.0),
            quantity: Quantity::from(5u32),
            aggressor_side: crate::core::types::Side::Ask,
            timestamp: 10000,
            buyer_order_id: None,
            seller_order_id: None,
        });

        let output = strategy.on_market_event(&other_trade, &context);
        assert!(output.metrics.is_none());
        assert!(output.orders.is_empty());
    }

    #[test]
    fn test_signal_deduplication() {
        let mut strategy = MeanReversionStrategy::new(
            "dedup_test".to_string(),
            1,
            MeanReversionConfig {
                lookback_period: 3,
                entry_threshold: 1.0,
                exit_threshold: 0.5,
                ..Default::default()
            },
        );

        let context = StrategyContext::new(
            "dedup_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        // Set up for signal generation
        strategy.current_mean = Some(100.0);
        strategy.current_std = Some(2.0);
        strategy.last_signal = Signal::None;

        // First signal should generate order
        let signal = strategy.generate_signal(97.5, 0); // Long signal
        assert_eq!(signal, Signal::Long);

        if let Some(order) = strategy.create_order(signal, Price::from_f64(97.5), &context) {
            strategy.last_signal = signal; // Update last signal
            assert_eq!(order.side, OrderSide::Buy);
        }

        // Same signal again should not generate new order (would be filtered in on_market_event)
        assert_eq!(strategy.last_signal, Signal::Long);

        // Different signal should generate order
        let new_signal = strategy.generate_signal(102.5, 0); // Short signal
        assert_eq!(new_signal, Signal::Short);
        assert_ne!(new_signal, strategy.last_signal); // Different from last
    }

    #[test]
    fn test_edge_cases_and_error_handling() {
        let mut strategy = MeanReversionStrategy::new(
            "edge_test".to_string(),
            1,
            MeanReversionConfig {
                lookback_period: 2,
                entry_threshold: 1.0,
                exit_threshold: 0.5,
                order_size: 0, // Edge case: zero order size
                ..Default::default()
            },
        );

        let context = StrategyContext::new(
            "edge_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        // Test with zero order size
        let order = strategy.create_order(Signal::Long, Price::from_f64(100.0), &context);
        assert!(order.is_none()); // Should not create order with zero quantity

        // Test with very small lookback period
        strategy.update_statistics(Price::from_f64(100.0), 1000);
        strategy.update_statistics(Price::from_f64(100.0), 2000);

        assert!(strategy.current_mean.is_some());
        assert!(strategy.current_std.is_some());
        assert_eq!(strategy.current_std.unwrap(), 0.0); // Zero variance with same prices

        // Test signal generation with zero std dev
        let signal = strategy.generate_signal(105.0, 0);
        assert_eq!(signal, Signal::None); // Should handle gracefully

        // Test with extreme position sizes
        let context_large_pos = StrategyContext::new(
            "edge_test".to_string(),
            1000,
            FeaturePosition {
                quantity: 1000, // Very large position
                ..Default::default()
            },
            RiskLimits::default(),
            true,
        );

        let strategy2 = MeanReversionStrategy::new(
            "edge_test2".to_string(),
            1,
            MeanReversionConfig {
                order_size: 50, // Large order size
                ..Default::default()
            },
        );

        let order =
            strategy2.create_order(Signal::ExitLong, Price::from_f64(100.0), &context_large_pos);
        assert!(order.is_some());
        let order = order.unwrap();
        // Should use min(position_size, order_size)
        assert_eq!(order.quantity, Quantity::from(50u32));

        // Test with extremely large prices
        strategy.current_mean = Some(1_000_000.0);
        strategy.current_std = Some(10_000.0);

        let signal = strategy.generate_signal(980_000.0, 0); // 2 std devs below mean
        assert_eq!(signal, Signal::Long);
    }

    #[test]
    fn test_metrics_generation() {
        let mut strategy = MeanReversionStrategy::new(
            "metrics_test".to_string(),
            1,
            MeanReversionConfig {
                lookback_period: 3,
                ..Default::default()
            },
        );

        // Set up statistics
        strategy.current_mean = Some(100.0);
        strategy.current_std = Some(2.5);

        let context = StrategyContext::new(
            "metrics_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        // Process a trade event that should generate metrics
        use crate::market_data::events::TradeEvent;
        let trade_event = MarketEvent::Trade(TradeEvent {
            instrument_id: 1,
            trade_id: 12345,
            price: Price::from_f64(103.0),
            quantity: Quantity::from(10u32),
            aggressor_side: crate::core::types::Side::Bid,
            timestamp: 2000,
            buyer_order_id: None,
            seller_order_id: None,
        });

        let output = strategy.on_market_event(&trade_event, &context);

        // Should have metrics
        assert!(output.metrics.is_some());
        let metrics = output.metrics.unwrap();

        // Should contain expected metric values
        // Z-score = (103 - 100) / 2.5 = 1.2
        let expected_z_score = (103.0 - 100.0) / 2.5;

        // Verify metrics (assuming the implementation adds these metrics)
        // Note: This depends on the actual metrics implementation
        assert_eq!(metrics.timestamp, 2000);
    }
}
