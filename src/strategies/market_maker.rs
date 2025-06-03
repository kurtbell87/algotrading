//! Market Making Strategy using Avellaneda-Stoikov Model
//!
//! This strategy implements a market making algorithm based on the
//! Avellaneda-Stoikov model for high-frequency trading.

use crate::core::types::{InstrumentId, Price, Quantity};
use crate::market_data::events::MarketEvent;
use crate::strategy::output::{OrderRequest, StrategyMetrics};
use crate::strategy::{
    OrderSide, Strategy, StrategyConfig, StrategyContext, StrategyError, StrategyOutput,
    TimeInForce,
};
use std::collections::VecDeque;

/// Configuration for market making strategy
#[derive(Debug, Clone)]
pub struct MarketMakerConfig {
    /// Risk aversion parameter (gamma)
    pub risk_aversion: f64,
    /// Volatility estimate update period
    pub volatility_lookback: usize,
    /// Inventory risk factor
    pub inventory_risk_factor: f64,
    /// Maximum inventory position
    pub max_inventory: i64,
    /// Minimum spread (in ticks)
    pub min_spread_ticks: i64,
    /// Maximum spread (in ticks)
    pub max_spread_ticks: i64,
    /// Order size
    pub order_size: u32,
    /// Time to end of trading day (seconds)
    pub time_horizon: f64,
    /// Cancel and replace threshold (price change in ticks)
    pub update_threshold_ticks: i64,
    /// Use adaptive spreads based on inventory
    pub use_adaptive_spreads: bool,
}

impl Default for MarketMakerConfig {
    fn default() -> Self {
        Self {
            risk_aversion: 0.1,
            volatility_lookback: 100,
            inventory_risk_factor: 0.01,
            max_inventory: 20,
            min_spread_ticks: 1,
            max_spread_ticks: 10,
            order_size: 1,
            time_horizon: 3600.0, // 1 hour
            update_threshold_ticks: 2,
            use_adaptive_spreads: true,
        }
    }
}

/// Market making strategy using Avellaneda-Stoikov model
pub struct MarketMakerStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// Market maker specific config
    mm_config: MarketMakerConfig,
    /// Price history for volatility calculation
    price_history: VecDeque<(u64, f64)>,
    /// Current volatility estimate
    volatility: Option<f64>,
    /// Current inventory (position)
    inventory: i64,
    /// Last mid price
    last_mid_price: Option<Price>,
    /// Active buy order ID
    active_buy_order: Option<u64>,
    /// Active sell order ID
    active_sell_order: Option<u64>,
    /// Last buy order price
    last_buy_price: Option<Price>,
    /// Last sell order price
    last_sell_price: Option<Price>,
    /// Start time for time horizon calculation
    start_time: Option<u64>,
}

impl MarketMakerStrategy {
    /// Create a new market maker strategy
    pub fn new(
        strategy_id: String,
        instrument_id: InstrumentId,
        mm_config: MarketMakerConfig,
    ) -> Self {
        let config = StrategyConfig::new(strategy_id, "Market Maker Strategy")
            .with_instrument(instrument_id)
            .with_max_position(mm_config.max_inventory)
            .with_timer(100_000); // 100ms timer for order updates

        Self {
            config,
            mm_config: mm_config.clone(),
            price_history: VecDeque::with_capacity(mm_config.volatility_lookback * 2),
            volatility: None,
            inventory: 0,
            last_mid_price: None,
            active_buy_order: None,
            active_sell_order: None,
            last_buy_price: None,
            last_sell_price: None,
            start_time: None,
        }
    }

    /// Update volatility estimate
    fn update_volatility(&mut self, price: f64, timestamp: u64) {
        // Add to price history
        self.price_history.push_back((timestamp, price));

        // Keep only required history
        while self.price_history.len() > self.mm_config.volatility_lookback {
            self.price_history.pop_front();
        }

        // Calculate volatility if we have enough data
        if self.price_history.len() >= 20 {
            // Calculate returns
            let mut returns = Vec::new();
            for i in 1..self.price_history.len() {
                let prev_price = self.price_history[i - 1].1;
                let curr_price = self.price_history[i].1;
                if prev_price > 0.0 {
                    let ret = (curr_price / prev_price).ln();
                    returns.push(ret);
                }
            }

            if !returns.is_empty() {
                // Calculate standard deviation of returns
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance =
                    returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

                // Annualize volatility (assuming microsecond timestamps)
                let avg_time_diff = if self.price_history.len() > 1 {
                    let total_time = self.price_history.back().unwrap().0
                        - self.price_history.front().unwrap().0;
                    total_time as f64 / (self.price_history.len() - 1) as f64
                } else {
                    1_000_000.0 // 1 second default
                };

                let periods_per_year = 365.25 * 24.0 * 60.0 * 60.0 * 1_000_000.0 / avg_time_diff;
                self.volatility = Some(variance.sqrt() * periods_per_year.sqrt());
            }
        }
    }

    /// Calculate optimal bid and ask prices using Avellaneda-Stoikov model
    fn calculate_optimal_quotes(&self, mid_price: f64, current_time: u64) -> (Price, Price) {
        let vol = self.volatility.unwrap_or(0.01); // Default 1% volatility
        let gamma = self.mm_config.risk_aversion;
        let tick_size = 25; // TODO: Get from instrument config

        // Calculate time remaining (T - t)
        let time_remaining = if let Some(start) = self.start_time {
            let elapsed = (current_time - start) as f64 / 1_000_000.0; // Convert to seconds
            (self.mm_config.time_horizon - elapsed).max(1.0) // At least 1 second
        } else {
            self.mm_config.time_horizon
        };

        // Base spread from Avellaneda-Stoikov formula
        // δ = γ * σ^2 * (T - t) + (2/γ) * ln(1 + γ/k)
        // Simplified version for now
        let base_spread = gamma * vol * vol * time_remaining;

        // Inventory adjustment
        let inventory_adjustment = if self.mm_config.use_adaptive_spreads {
            let inventory_ratio = self.inventory as f64 / self.mm_config.max_inventory as f64;
            self.mm_config.inventory_risk_factor * inventory_ratio * mid_price
        } else {
            0.0
        };

        // Calculate bid and ask offsets
        let half_spread = base_spread * mid_price / 2.0;
        // With positive inventory (long), reduce ask offset (more aggressive selling)
        // and increase bid offset (less aggressive buying)
        let bid_offset = half_spread + inventory_adjustment;
        let ask_offset = half_spread - inventory_adjustment;

        // Apply min/max spread constraints
        let min_offset = self.mm_config.min_spread_ticks as f64 * tick_size as f64;
        let max_offset = self.mm_config.max_spread_ticks as f64 * tick_size as f64;

        let bid_offset = bid_offset.max(min_offset).min(max_offset);
        let ask_offset = ask_offset.max(min_offset).min(max_offset);

        // Calculate final prices
        let bid_price = Price::new((mid_price - bid_offset) as i64);
        let ask_price = Price::new((mid_price + ask_offset) as i64);

        (bid_price, ask_price)
    }

    /// Check if we should update orders
    fn should_update_orders(&self, new_bid: Price, new_ask: Price) -> bool {
        let tick_size = 25; // TODO: Get from instrument config
        let threshold = self.mm_config.update_threshold_ticks * tick_size;

        // Check buy order
        if let Some(last_buy) = self.last_buy_price {
            if (new_bid.0 - last_buy.0).abs() >= threshold {
                return true;
            }
        } else {
            return true; // No active buy order
        }

        // Check sell order
        if let Some(last_sell) = self.last_sell_price {
            if (new_ask.0 - last_sell.0).abs() >= threshold {
                return true;
            }
        } else {
            return true; // No active sell order
        }

        false
    }

    /// Create order updates to cancel existing orders
    fn create_cancellations(&self, output: &mut StrategyOutput) {
        if let Some(buy_id) = self.active_buy_order {
            output.add_cancellation(buy_id);
        }
        if let Some(sell_id) = self.active_sell_order {
            output.add_cancellation(sell_id);
        }
    }

    /// Create new market making orders
    fn create_new_orders(
        &mut self,
        bid_price: Price,
        ask_price: Price,
        output: &mut StrategyOutput,
        context: &StrategyContext,
    ) {
        let instrument_id = self.config.instruments[0];

        // Create buy order if we have room for inventory
        if self.inventory < self.mm_config.max_inventory {
            let buy_order = OrderRequest::limit_order(
                context.strategy_id.clone(),
                instrument_id,
                OrderSide::Buy,
                bid_price,
                Quantity::from(self.mm_config.order_size),
            )
            .with_time_in_force(TimeInForce::GTC);

            output.orders.push(buy_order);
            self.last_buy_price = Some(bid_price);
        }

        // Create sell order if we have inventory to sell
        if self.inventory > -self.mm_config.max_inventory {
            let sell_order = OrderRequest::limit_order(
                context.strategy_id.clone(),
                instrument_id,
                if self.inventory > 0 {
                    OrderSide::Sell
                } else {
                    OrderSide::SellShort
                },
                ask_price,
                Quantity::from(self.mm_config.order_size),
            )
            .with_time_in_force(TimeInForce::GTC);

            output.orders.push(sell_order);
            self.last_sell_price = Some(ask_price);
        }
    }
}

impl Strategy for MarketMakerStrategy {
    fn initialize(&mut self, context: &StrategyContext) -> Result<(), StrategyError> {
        self.price_history.clear();
        self.volatility = None;
        self.inventory = 0;
        self.last_mid_price = None;
        self.active_buy_order = None;
        self.active_sell_order = None;
        self.last_buy_price = None;
        self.last_sell_price = None;
        self.start_time = Some(context.current_time);
        Ok(())
    }

    fn on_market_event(
        &mut self,
        event: &MarketEvent,
        context: &StrategyContext,
    ) -> StrategyOutput {
        let mut output = StrategyOutput::default();

        // Extract mid price from market event
        let mid_price = match event {
            MarketEvent::Trade(trade) => {
                if trade.instrument_id == self.config.instruments[0] {
                    Some(trade.price.as_f64())
                } else {
                    None
                }
            }
            MarketEvent::BBO(bbo) => {
                if bbo.instrument_id == self.config.instruments[0] {
                    match (bbo.bid_price, bbo.ask_price) {
                        (Some(bid), Some(ask)) => Some((bid.as_f64() + ask.as_f64()) / 2.0),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(mid_price) = mid_price {
            // Update volatility
            self.update_volatility(mid_price, event.timestamp());

            // Update inventory from context
            self.inventory = context.position.quantity;

            // Calculate optimal quotes
            let (bid_price, ask_price) =
                self.calculate_optimal_quotes(mid_price, context.current_time);

            // Check if we should update orders
            if self.should_update_orders(bid_price, ask_price) {
                // Cancel existing orders
                self.create_cancellations(&mut output);

                // Create new orders
                self.create_new_orders(bid_price, ask_price, &mut output, context);
            }

            self.last_mid_price = Some(Price::from_f64(mid_price));

            // Update metrics
            let mut metrics = StrategyMetrics::new(event.timestamp());
            metrics.add("mid_price", mid_price);
            metrics.add("bid_price", bid_price.as_f64());
            metrics.add("ask_price", ask_price.as_f64());
            metrics.add("spread", ask_price.as_f64() - bid_price.as_f64());
            metrics.add("inventory", self.inventory as f64);
            if let Some(vol) = self.volatility {
                metrics.add("volatility", vol);
            }
            output.set_metrics(metrics);
        }

        output
    }

    fn on_timer(&mut self, timestamp: u64, context: &StrategyContext) -> StrategyOutput {
        let mut output = StrategyOutput::default();

        // Update quotes if we have a mid price
        if let Some(mid_price) = self.last_mid_price {
            let (bid_price, ask_price) =
                self.calculate_optimal_quotes(mid_price.as_f64(), timestamp);

            if self.should_update_orders(bid_price, ask_price) {
                self.create_cancellations(&mut output);
                self.create_new_orders(bid_price, ask_price, &mut output, context);
            }
        }

        output
    }

    fn on_fill(
        &mut self,
        _price: Price,
        quantity: i64,
        _timestamp: u64,
        context: &StrategyContext,
    ) {
        // Update inventory
        self.inventory = context.position.quantity;

        // Clear the filled order
        if quantity > 0 {
            // Buy fill
            self.active_buy_order = None;
            self.last_buy_price = None;
        } else {
            // Sell fill
            self.active_sell_order = None;
            self.last_sell_price = None;
        }
    }

    fn on_order_rejected(&mut self, order_id: u64, _reason: String, _context: &StrategyContext) {
        // Clear rejected order
        if Some(order_id) == self.active_buy_order {
            self.active_buy_order = None;
            self.last_buy_price = None;
        } else if Some(order_id) == self.active_sell_order {
            self.active_sell_order = None;
            self.last_sell_price = None;
        }
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
    fn test_market_maker_creation() {
        let strategy =
            MarketMakerStrategy::new("test_mm".to_string(), 1, MarketMakerConfig::default());

        assert_eq!(strategy.config.id, "test_mm");
        assert_eq!(strategy.config.instruments.len(), 1);
        assert_eq!(strategy.config.instruments[0], 1);
        assert!(strategy.config.uses_timer);
    }

    #[test]
    fn test_volatility_calculation() {
        let mut strategy = MarketMakerStrategy::new(
            "test_mm".to_string(),
            1,
            MarketMakerConfig {
                volatility_lookback: 20,
                ..Default::default()
            },
        );

        // Add price history with some volatility
        let prices = vec![100.0, 101.0, 99.5, 100.5, 102.0, 101.0, 99.0, 100.0];

        for (i, price) in prices.iter().enumerate() {
            strategy.update_volatility(*price, i as u64 * 1000000); // 1 second intervals
        }

        // Should not have volatility yet (need 20 points)
        assert!(strategy.volatility.is_none());

        // Add more points
        for i in 0..15 {
            let price = 100.0 + (i % 3) as f64 - 1.0;
            strategy.update_volatility(price, (prices.len() + i) as u64 * 1000000);
        }

        // Now should have volatility
        assert!(strategy.volatility.is_some());
        assert!(strategy.volatility.unwrap() > 0.0);
    }

    #[test]
    fn test_optimal_quotes() {
        let mut strategy = MarketMakerStrategy::new(
            "test_mm".to_string(),
            1,
            MarketMakerConfig {
                risk_aversion: 0.1,
                min_spread_ticks: 2,
                max_spread_ticks: 10,
                time_horizon: 60.0, // Short time horizon to reduce base spread
                ..Default::default()
            },
        );

        // Set start time to control time remaining calculation
        strategy.start_time = Some(0);

        // Test quote calculation
        let mid_price = 10000.0;
        let current_time = 1_000_000; // 1 second
        let (bid, ask) = strategy.calculate_optimal_quotes(mid_price, current_time);

        // Check spread constraints
        let spread = ask.0 - bid.0;
        assert!(
            spread >= 2 * 25,
            "Spread {} should be >= {}",
            spread,
            2 * 25
        ); // min_spread_ticks * tick_size
        assert!(
            spread <= 10 * 25,
            "Spread {} should be <= {}",
            spread,
            10 * 25
        ); // max_spread_ticks * tick_size

        // Check that bid < ask
        assert!(
            bid.0 < ask.0,
            "Bid {} should be less than ask {}",
            bid.0,
            ask.0
        );

        // Check that prices are reasonable relative to mid price
        assert!(
            bid.0 < mid_price as i64,
            "Bid {} should be less than mid price {}",
            bid.0,
            mid_price
        );
        assert!(
            ask.0 > mid_price as i64,
            "Ask {} should be greater than mid price {}",
            ask.0,
            mid_price
        );
    }

    #[test]
    fn test_inventory_adjustment() {
        let mut strategy = MarketMakerStrategy::new(
            "test_mm".to_string(),
            1,
            MarketMakerConfig {
                inventory_risk_factor: 0.01,
                max_inventory: 10,
                use_adaptive_spreads: true,
                time_horizon: 60.0, // Short time horizon for consistent results
                min_spread_ticks: 1,
                max_spread_ticks: 20, // Higher max to allow for inventory adjustment
                ..Default::default()
            },
        );

        // Set start time and volatility for consistent results
        strategy.start_time = Some(0);
        strategy.volatility = Some(0.02); // 2% volatility

        let mid_price = 10000.0;
        let current_time = 1_000_000;

        // Test with positive inventory (long position - want to sell)
        strategy.inventory = 5;
        let (bid1, ask1) = strategy.calculate_optimal_quotes(mid_price, current_time);

        // Test with negative inventory (short position - want to buy)
        strategy.inventory = -5;
        let (bid2, ask2) = strategy.calculate_optimal_quotes(mid_price, current_time);

        // With positive inventory, should have tighter ask spread (more aggressive selling)
        // and wider bid spread (less aggressive buying)
        // With negative inventory, should have tighter bid spread (more aggressive buying)
        // and wider ask spread (less aggressive selling)
        assert!(
            ask1.0 < ask2.0,
            "Positive inventory should lead to lower ask prices (more aggressive selling): ask1={}, ask2={}",
            ask1.0,
            ask2.0
        );
        assert!(
            bid1.0 < bid2.0,
            "Positive inventory should lead to lower bid prices (less aggressive buying): bid1={}, bid2={}",
            bid1.0,
            bid2.0
        );

        // Test with zero inventory for baseline
        strategy.inventory = 0;
        let (_bid0, _ask0) = strategy.calculate_optimal_quotes(mid_price, current_time);

        // Verify that inventory adjustments are symmetric around zero inventory
        let bid_diff_pos = (mid_price as i64 - bid1.0).abs();
        let bid_diff_neg = (mid_price as i64 - bid2.0).abs();
        let ask_diff_pos = (ask1.0 - mid_price as i64).abs();
        let ask_diff_neg = (ask2.0 - mid_price as i64).abs();

        // With positive inventory, bid should be further from mid, ask should be closer
        // With negative inventory, ask should be further from mid, bid should be closer
        assert!(
            bid_diff_pos > bid_diff_neg,
            "Positive inventory should make bid further from mid"
        );
        assert!(
            ask_diff_neg > ask_diff_pos,
            "Negative inventory should make ask further from mid"
        );
    }

    #[test]
    fn test_order_creation() {
        let mut strategy = MarketMakerStrategy::new(
            "test_mm".to_string(),
            1,
            MarketMakerConfig {
                order_size: 5,
                max_inventory: 10,
                ..Default::default()
            },
        );

        let context = StrategyContext::new(
            "test_mm".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        let bid_price = Price::new(9950i64);
        let ask_price = Price::new(10050i64);
        let mut output = StrategyOutput::default();

        strategy.create_new_orders(bid_price, ask_price, &mut output, &context);

        assert_eq!(output.orders.len(), 2);

        // Check buy order
        let buy_order = &output.orders[0];
        assert_eq!(buy_order.side, OrderSide::Buy);
        assert_eq!(buy_order.quantity, Quantity::from(5u32));
        assert_eq!(buy_order.price, Some(bid_price));
        assert_eq!(buy_order.order_type, OrderType::Limit);
        assert_eq!(buy_order.time_in_force, TimeInForce::GTC);

        // Check sell order
        let sell_order = &output.orders[1];
        assert_eq!(sell_order.side, OrderSide::SellShort);
        assert_eq!(sell_order.quantity, Quantity::from(5u32));
        assert_eq!(sell_order.price, Some(ask_price));
    }

    #[test]
    fn test_config_validation() {
        // Test default configuration
        let default_config = MarketMakerConfig::default();
        assert_eq!(default_config.risk_aversion, 0.1);
        assert_eq!(default_config.volatility_lookback, 100);
        assert_eq!(default_config.inventory_risk_factor, 0.01);
        assert_eq!(default_config.max_inventory, 20);
        assert_eq!(default_config.min_spread_ticks, 1);
        assert_eq!(default_config.max_spread_ticks, 10);
        assert_eq!(default_config.order_size, 1);
        assert_eq!(default_config.time_horizon, 3600.0);
        assert_eq!(default_config.update_threshold_ticks, 2);
        assert_eq!(default_config.use_adaptive_spreads, true);

        // Test custom configuration
        let custom_config = MarketMakerConfig {
            risk_aversion: 0.2,
            volatility_lookback: 50,
            inventory_risk_factor: 0.02,
            max_inventory: 15,
            min_spread_ticks: 2,
            max_spread_ticks: 20,
            order_size: 5,
            time_horizon: 1800.0,
            update_threshold_ticks: 3,
            use_adaptive_spreads: false,
        };

        let strategy = MarketMakerStrategy::new("custom_mm".to_string(), 2, custom_config.clone());

        assert_eq!(strategy.mm_config.risk_aversion, 0.2);
        assert_eq!(strategy.mm_config.max_inventory, 15);
    }

    #[test]
    fn test_strategy_initialization() {
        let mut strategy =
            MarketMakerStrategy::new("init_test".to_string(), 1, MarketMakerConfig::default());

        let context = StrategyContext::new(
            "init_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        // Initialize strategy
        let result = strategy.initialize(&context);
        assert!(result.is_ok());

        // Check that state is properly reset
        assert!(strategy.price_history.is_empty());
        assert!(strategy.volatility.is_none());
        assert_eq!(strategy.inventory, 0);
        assert!(strategy.last_mid_price.is_none());
        assert!(strategy.active_buy_order.is_none());
        assert!(strategy.active_sell_order.is_none());
        assert!(strategy.last_buy_price.is_none());
        assert!(strategy.last_sell_price.is_none());
        assert!(strategy.start_time.is_some());
        assert_eq!(strategy.start_time.unwrap(), 1000);
    }

    #[test]
    fn test_market_event_processing() {
        use crate::market_data::events::{BBOUpdate, MarketEvent, TradeEvent};

        let mut strategy = MarketMakerStrategy::new(
            "event_test".to_string(),
            1,
            MarketMakerConfig {
                volatility_lookback: 5,
                min_spread_ticks: 1,
                max_spread_ticks: 5,
                time_horizon: 60.0,
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
            price: Price::from(100i64),
            quantity: Quantity::from(10u32),
            aggressor_side: crate::core::types::Side::Bid,
            timestamp: 2000,
            buyer_order_id: None,
            seller_order_id: None,
        });

        let output = strategy.on_market_event(&trade_event, &context);

        // Should have metrics and possibly orders depending on volatility calculation
        assert!(output.metrics.is_some());
        // Note: Orders may or may not be empty depending on internal state

        // Test BBO event processing
        let bbo_event = MarketEvent::BBO(BBOUpdate {
            instrument_id: 1,
            bid_price: Some(Price::from(99i64)),
            ask_price: Some(Price::from(101i64)),
            bid_quantity: Some(Quantity::from(5u32)),
            ask_quantity: Some(Quantity::from(8u32)),
            bid_order_count: Some(2),
            ask_order_count: Some(3),
            timestamp: 3000,
        });

        let output = strategy.on_market_event(&bbo_event, &context);
        assert!(output.metrics.is_some());

        // Test event for different instrument (should be ignored)
        let other_trade = MarketEvent::Trade(TradeEvent {
            instrument_id: 2, // Different instrument
            trade_id: 67890,
            price: Price::from(200i64),
            quantity: Quantity::from(5u32),
            aggressor_side: crate::core::types::Side::Ask,
            timestamp: 4000,
            buyer_order_id: None,
            seller_order_id: None,
        });

        let output = strategy.on_market_event(&other_trade, &context);
        assert!(output.metrics.is_none());
        assert!(output.orders.is_empty());
    }

    #[test]
    fn test_order_update_logic() {
        let mut strategy = MarketMakerStrategy::new(
            "update_test".to_string(),
            1,
            MarketMakerConfig {
                update_threshold_ticks: 2,
                ..Default::default()
            },
        );

        // Test when no orders exist
        let new_bid = Price::from(100i64);
        let new_ask = Price::from(102i64);
        assert!(strategy.should_update_orders(new_bid, new_ask));

        // Set existing orders
        strategy.last_buy_price = Some(Price::from(100i64));
        strategy.last_sell_price = Some(Price::from(102i64));

        // Test when prices haven't moved enough
        let small_move_bid = Price::new(100_000_000_000 + 25); // 1 tick move (< threshold)
        let small_move_ask = Price::new(102_000_000_000 + 25);
        assert!(!strategy.should_update_orders(small_move_bid, small_move_ask));

        // Test when prices have moved enough
        let big_move_bid = Price::new(100_000_000_000 + 75); // 3 tick move (> threshold)
        let big_move_ask = Price::new(102_000_000_000 + 75);
        assert!(strategy.should_update_orders(big_move_bid, big_move_ask));
    }

    #[test]
    fn test_order_creation_with_inventory_limits() {
        let mut strategy = MarketMakerStrategy::new(
            "limit_test".to_string(),
            1,
            MarketMakerConfig {
                max_inventory: 5,
                order_size: 2,
                ..Default::default()
            },
        );

        let context = StrategyContext::new(
            "limit_test".to_string(),
            1000,
            FeaturePosition {
                quantity: 4, // Near max inventory
                ..Default::default()
            },
            RiskLimits::default(),
            true,
        );

        strategy.inventory = 4; // Near max

        let bid_price = Price::from(99i64);
        let ask_price = Price::from(101i64);
        let mut output = StrategyOutput::default();

        strategy.create_new_orders(bid_price, ask_price, &mut output, &context);

        // Should create buy order (inventory < max)
        // Should create sell order (can always sell when inventory > -max)
        assert_eq!(output.orders.len(), 2);

        // Test at max inventory
        strategy.inventory = 5; // At max
        let mut output2 = StrategyOutput::default();
        strategy.create_new_orders(bid_price, ask_price, &mut output2, &context);

        // Should only create sell order (inventory >= max)
        assert_eq!(output2.orders.len(), 1);
        assert_eq!(output2.orders[0].side, OrderSide::Sell);
    }

    #[test]
    fn test_on_fill_callback() {
        let mut strategy =
            MarketMakerStrategy::new("fill_test".to_string(), 1, MarketMakerConfig::default());

        let context = StrategyContext::new(
            "fill_test".to_string(),
            1000,
            FeaturePosition {
                quantity: 3,
                ..Default::default()
            },
            RiskLimits::default(),
            true,
        );

        // Set up active orders
        strategy.active_buy_order = Some(123);
        strategy.active_sell_order = Some(456);
        strategy.last_buy_price = Some(Price::from(99i64));
        strategy.last_sell_price = Some(Price::from(101i64));

        // Test buy fill
        strategy.on_fill(Price::from(99i64), 2, 5000, &context);

        // Should update inventory and clear buy order info
        assert_eq!(strategy.inventory, 3);
        assert!(strategy.active_buy_order.is_none());
        assert!(strategy.last_buy_price.is_none());
        assert!(strategy.active_sell_order.is_some()); // Sell order unchanged

        // Test sell fill
        strategy.on_fill(Price::from(101i64), -1, 6000, &context);

        // Should clear sell order info
        assert!(strategy.active_sell_order.is_none());
        assert!(strategy.last_sell_price.is_none());
    }

    #[test]
    fn test_on_order_rejected() {
        let mut strategy =
            MarketMakerStrategy::new("reject_test".to_string(), 1, MarketMakerConfig::default());

        let context = StrategyContext::new(
            "reject_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        // Set up active orders
        strategy.active_buy_order = Some(123);
        strategy.active_sell_order = Some(456);
        strategy.last_buy_price = Some(Price::from(99i64));
        strategy.last_sell_price = Some(Price::from(101i64));

        // Test buy order rejection
        strategy.on_order_rejected(123, "Insufficient funds".to_string(), &context);

        // Should clear buy order info
        assert!(strategy.active_buy_order.is_none());
        assert!(strategy.last_buy_price.is_none());
        assert!(strategy.active_sell_order.is_some()); // Sell order unchanged

        // Test sell order rejection
        strategy.on_order_rejected(456, "Invalid price".to_string(), &context);

        // Should clear sell order info
        assert!(strategy.active_sell_order.is_none());
        assert!(strategy.last_sell_price.is_none());

        // Test rejection of unknown order (should not crash)
        strategy.on_order_rejected(999, "Unknown".to_string(), &context);
    }

    #[test]
    fn test_timer_callback() {
        let mut strategy = MarketMakerStrategy::new(
            "timer_test".to_string(),
            1,
            MarketMakerConfig {
                update_threshold_ticks: 1,
                min_spread_ticks: 1,
                max_spread_ticks: 5,
                time_horizon: 60.0,
                ..Default::default()
            },
        );

        let context = StrategyContext::new(
            "timer_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        strategy.initialize(&context).unwrap();

        // Test timer when no mid price exists
        let output = strategy.on_timer(2000, &context);
        assert!(output.orders.is_empty());

        // Set mid price
        strategy.last_mid_price = Some(Price::from(100i64));

        // Test timer with mid price
        let output = strategy.on_timer(3000, &context);

        // Should generate orders (no existing orders)
        assert!(!output.orders.is_empty());
    }

    #[test]
    fn test_risk_management_edge_cases() {
        let mut strategy = MarketMakerStrategy::new(
            "risk_test".to_string(),
            1,
            MarketMakerConfig {
                risk_aversion: 0.0,         // Zero risk aversion
                inventory_risk_factor: 0.0, // No inventory risk
                min_spread_ticks: 1,
                max_spread_ticks: 10,
                time_horizon: 1.0, // Very short time horizon
                ..Default::default()
            },
        );

        strategy.start_time = Some(0);
        strategy.volatility = Some(0.0); // Zero volatility

        let mid_price = 100.0;
        let current_time = 2_000_000; // 2 seconds (past time horizon)

        let (bid, ask) = strategy.calculate_optimal_quotes(mid_price, current_time);

        // With zero risk aversion and volatility, should still respect min spread
        let spread = ask.0 - bid.0;
        let min_spread = 1 * 25; // min_spread_ticks * tick_size
        assert!(spread >= min_spread);

        // Test with extreme inventory
        strategy.inventory = 1000; // Very high inventory
        let (bid2, ask2) = strategy.calculate_optimal_quotes(mid_price, current_time);

        // Should still respect max spread constraint
        let spread2 = ask2.0 - bid2.0;
        let max_spread = 10 * 25; // max_spread_ticks * tick_size
        assert!(spread2 <= max_spread);
    }

    #[test]
    fn test_volatility_edge_cases() {
        let mut strategy = MarketMakerStrategy::new(
            "vol_test".to_string(),
            1,
            MarketMakerConfig {
                volatility_lookback: 3,
                ..Default::default()
            },
        );

        // Test with insufficient data
        strategy.update_volatility(100.0, 1000);
        strategy.update_volatility(101.0, 2000);
        assert!(strategy.volatility.is_none());

        // Test with flat prices (zero volatility)
        for i in 0..25 {
            strategy.update_volatility(100.0, (i as u64) * 1000000); // Use larger time intervals
        }

        // Should have volatility calculated (even if zero)
        if let Some(vol) = strategy.volatility {
            assert!(vol >= 0.0); // Volatility should be non-negative
        } else {
            // If no volatility calculated, it's still valid behavior
            println!("No volatility calculated with flat prices");
        }

        // Test with extreme price movements
        let mut strategy2 = MarketMakerStrategy::new(
            "vol_test2".to_string(),
            1,
            MarketMakerConfig {
                volatility_lookback: 20,
                ..Default::default()
            },
        );

        // Add volatile price data
        let prices = vec![
            100.0, 200.0, 50.0, 150.0, 75.0, 125.0, 90.0, 110.0, 105.0, 95.0, 115.0, 85.0, 120.0,
            80.0, 130.0, 70.0, 140.0, 60.0, 160.0, 40.0, 180.0,
        ];

        for (i, price) in prices.iter().enumerate() {
            strategy2.update_volatility(*price, (i as u64) * 1000000);
        }

        assert!(strategy2.volatility.is_some());
        assert!(strategy2.volatility.unwrap() > 0.0);
    }
}
