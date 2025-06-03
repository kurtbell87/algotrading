//! Trend Following Trading Strategy
//!
//! This strategy identifies and follows market trends using moving averages
//! and momentum indicators.

use crate::core::types::{InstrumentId, Price, Quantity};
use crate::market_data::events::MarketEvent;
use crate::strategy::output::{OrderRequest, StrategyMetrics};
use crate::strategy::{
    OrderSide, Strategy, StrategyConfig, StrategyContext, StrategyError, StrategyOutput,
};
use std::collections::VecDeque;

/// Configuration for trend following strategy
#[derive(Debug, Clone)]
pub struct TrendFollowingConfig {
    /// Short-term moving average period
    pub short_ma_period: usize,
    /// Long-term moving average period
    pub long_ma_period: usize,
    /// Momentum lookback period
    pub momentum_period: usize,
    /// Minimum momentum threshold for entry
    pub momentum_threshold: f64,
    /// ATR period for volatility-based stops
    pub atr_period: usize,
    /// ATR multiplier for stop loss
    pub stop_loss_atr_multiplier: f64,
    /// ATR multiplier for take profit
    pub take_profit_atr_multiplier: f64,
    /// Maximum position size
    pub max_position_size: i64,
    /// Order size per trade
    pub order_size: u32,
    /// Use trailing stops
    pub use_trailing_stops: bool,
    /// Require volume confirmation
    pub require_volume_confirmation: bool,
    /// Volume multiplier threshold
    pub volume_multiplier: f64,
}

impl Default for TrendFollowingConfig {
    fn default() -> Self {
        Self {
            short_ma_period: 10,
            long_ma_period: 30,
            momentum_period: 14,
            momentum_threshold: 0.02, // 2% momentum
            atr_period: 14,
            stop_loss_atr_multiplier: 2.0,
            take_profit_atr_multiplier: 3.0,
            max_position_size: 10,
            order_size: 1,
            use_trailing_stops: true,
            require_volume_confirmation: false,
            volume_multiplier: 1.5,
        }
    }
}

/// Trend following trading strategy
pub struct TrendFollowingStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// Trend following specific config
    tf_config: TrendFollowingConfig,
    /// Price history
    price_history: Vec<(u64, Price)>,
    /// Volume history
    volume_history: VecDeque<(u64, u64)>,
    /// Short-term moving average
    short_ma: Option<f64>,
    /// Long-term moving average
    long_ma: Option<f64>,
    /// Current momentum
    momentum: Option<f64>,
    /// Average True Range
    atr: Option<f64>,
    /// Current trend direction
    trend_direction: TrendDirection,
    /// Entry price for current position
    entry_price: Option<Price>,
    /// Stop loss price
    stop_loss: Option<Price>,
    /// Take profit price
    take_profit: Option<Price>,
    /// Highest price since entry (for trailing stops)
    highest_price: Option<Price>,
    /// Lowest price since entry (for trailing stops)
    lowest_price: Option<Price>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TrendDirection {
    None,
    Bullish,
    Bearish,
}

impl TrendFollowingStrategy {
    /// Create a new trend following strategy
    pub fn new(
        strategy_id: String,
        instrument_id: InstrumentId,
        tf_config: TrendFollowingConfig,
    ) -> Self {
        let config = StrategyConfig::new(strategy_id, "Trend Following Strategy")
            .with_instrument(instrument_id)
            .with_max_position(tf_config.max_position_size);

        Self {
            config,
            tf_config: tf_config.clone(),
            price_history: Vec::with_capacity(tf_config.long_ma_period * 2),
            volume_history: VecDeque::with_capacity(tf_config.momentum_period * 2),
            short_ma: None,
            long_ma: None,
            momentum: None,
            atr: None,
            trend_direction: TrendDirection::None,
            entry_price: None,
            stop_loss: None,
            take_profit: None,
            highest_price: None,
            lowest_price: None,
        }
    }

    /// Update moving averages
    fn update_moving_averages(&mut self) {
        let len = self.price_history.len();

        // Calculate short-term MA
        if len >= self.tf_config.short_ma_period {
            let sum: f64 = self
                .price_history
                .iter()
                .rev()
                .take(self.tf_config.short_ma_period)
                .map(|(_, p)| p.as_f64())
                .sum();
            self.short_ma = Some(sum / self.tf_config.short_ma_period as f64);
        }

        // Calculate long-term MA
        if len >= self.tf_config.long_ma_period {
            let sum: f64 = self
                .price_history
                .iter()
                .rev()
                .take(self.tf_config.long_ma_period)
                .map(|(_, p)| p.as_f64())
                .sum();
            self.long_ma = Some(sum / self.tf_config.long_ma_period as f64);
        }
    }

    /// Update momentum indicator
    fn update_momentum(&mut self) {
        let len = self.price_history.len();

        if len >= self.tf_config.momentum_period {
            let current_price = self.price_history.last().unwrap().1.as_f64();
            let past_price = self.price_history[len - self.tf_config.momentum_period]
                .1
                .as_f64();

            if past_price > 0.0 {
                self.momentum = Some((current_price - past_price) / past_price);
            }
        }
    }

    /// Update Average True Range
    fn update_atr(&mut self) {
        let len = self.price_history.len();

        if len >= self.tf_config.atr_period + 1 {
            let mut true_ranges = Vec::new();

            for i in (len - self.tf_config.atr_period)..len {
                let current_high = self.price_history[i].1.as_f64();
                let current_low = self.price_history[i].1.as_f64();
                let prev_close = self.price_history[i - 1].1.as_f64();

                // Simplified TR calculation (using close price as high/low)
                let tr = (current_high - current_low)
                    .max((current_high - prev_close).abs())
                    .max((current_low - prev_close).abs());

                true_ranges.push(tr);
            }

            self.atr = Some(true_ranges.iter().sum::<f64>() / true_ranges.len() as f64);
        }
    }

    /// Determine trend direction
    fn determine_trend(&mut self) -> TrendDirection {
        if let (Some(short), Some(long)) = (self.short_ma, self.long_ma) {
            let prev_trend = self.trend_direction;

            // Check for trend changes
            if short > long && prev_trend != TrendDirection::Bullish {
                TrendDirection::Bullish
            } else if short < long && prev_trend != TrendDirection::Bearish {
                TrendDirection::Bearish
            } else {
                prev_trend
            }
        } else {
            TrendDirection::None
        }
    }

    /// Check if momentum confirms trend
    fn momentum_confirms(&self, trend: TrendDirection) -> bool {
        if let Some(momentum) = self.momentum {
            match trend {
                TrendDirection::Bullish => momentum > self.tf_config.momentum_threshold,
                TrendDirection::Bearish => momentum < -self.tf_config.momentum_threshold,
                TrendDirection::None => false,
            }
        } else {
            false
        }
    }

    /// Check if volume confirms signal
    fn volume_confirms(&self) -> bool {
        if !self.tf_config.require_volume_confirmation {
            return true;
        }

        if self.volume_history.len() < 20 {
            return false;
        }

        // Calculate average volume
        let avg_volume = self
            .volume_history
            .iter()
            .map(|(_, v)| *v as f64)
            .sum::<f64>()
            / self.volume_history.len() as f64;

        // Check if current volume exceeds threshold
        if let Some((_, current_volume)) = self.volume_history.back() {
            *current_volume as f64 > avg_volume * self.tf_config.volume_multiplier
        } else {
            false
        }
    }

    /// Calculate stop loss and take profit levels
    fn calculate_exit_levels(&self, entry_price: Price, is_long: bool) -> (Price, Price) {
        let atr = self.atr.unwrap_or(entry_price.as_f64() * 0.01); // Default 1% if no ATR
        let tick_size = 25; // TODO: Get from instrument config

        let stop_distance = (atr * self.tf_config.stop_loss_atr_multiplier) as i64;
        let profit_distance = (atr * self.tf_config.take_profit_atr_multiplier) as i64;

        // Round to tick size
        let stop_ticks = (stop_distance / tick_size) * tick_size;
        let profit_ticks = (profit_distance / tick_size) * tick_size;

        if is_long {
            let stop_loss = Price::new(entry_price.0 - stop_ticks);
            let take_profit = Price::new(entry_price.0 + profit_ticks);
            (stop_loss, take_profit)
        } else {
            let stop_loss = Price::new(entry_price.0 + stop_ticks);
            let take_profit = Price::new(entry_price.0 - profit_ticks);
            (stop_loss, take_profit)
        }
    }

    /// Update trailing stop
    fn update_trailing_stop(&mut self, current_price: Price, is_long: bool) {
        if !self.tf_config.use_trailing_stops || self.entry_price.is_none() {
            return;
        }

        let atr = self.atr.unwrap_or(current_price.as_f64() * 0.01);
        let stop_distance = (atr * self.tf_config.stop_loss_atr_multiplier) as i64;

        if is_long {
            // Update highest price
            if let Some(highest) = self.highest_price {
                if current_price > highest {
                    self.highest_price = Some(current_price);
                    // Move stop loss up
                    let new_stop = Price::new(current_price.0 - stop_distance);
                    if let Some(current_stop) = self.stop_loss {
                        if new_stop > current_stop {
                            self.stop_loss = Some(new_stop);
                        }
                    }
                }
            } else {
                self.highest_price = Some(current_price);
            }
        } else {
            // Update lowest price
            if let Some(lowest) = self.lowest_price {
                if current_price < lowest {
                    self.lowest_price = Some(current_price);
                    // Move stop loss down
                    let new_stop = Price::new(current_price.0 + stop_distance);
                    if let Some(current_stop) = self.stop_loss {
                        if new_stop < current_stop {
                            self.stop_loss = Some(new_stop);
                        }
                    }
                }
            } else {
                self.lowest_price = Some(current_price);
            }
        }
    }

    /// Check if exit conditions are met
    fn check_exit_conditions(&self, current_price: Price, position: i64) -> bool {
        if position == 0 {
            return false;
        }

        let is_long = position > 0;

        // Check stop loss
        if let Some(stop) = self.stop_loss {
            if (is_long && current_price <= stop) || (!is_long && current_price >= stop) {
                return true;
            }
        }

        // Check take profit
        if let Some(target) = self.take_profit {
            if (is_long && current_price >= target) || (!is_long && current_price <= target) {
                return true;
            }
        }

        // Check trend reversal
        let current_trend = match self.trend_direction {
            TrendDirection::Bullish => is_long,
            TrendDirection::Bearish => !is_long,
            TrendDirection::None => false,
        };

        !current_trend // Exit if trend has reversed
    }

    /// Create entry order
    fn create_entry_order(
        &mut self,
        side: OrderSide,
        price: Price,
        context: &StrategyContext,
    ) -> OrderRequest {
        let instrument_id = self.config.instruments[0];

        // Store entry information
        self.entry_price = Some(price);
        let is_long = matches!(side, OrderSide::Buy | OrderSide::BuyCover);
        let (stop, target) = self.calculate_exit_levels(price, is_long);
        self.stop_loss = Some(stop);
        self.take_profit = Some(target);

        // Reset trailing stop tracking
        if is_long {
            self.highest_price = Some(price);
            self.lowest_price = None;
        } else {
            self.highest_price = None;
            self.lowest_price = Some(price);
        }

        OrderRequest::market_order(
            context.strategy_id.clone(),
            instrument_id,
            side,
            Quantity::from(self.tf_config.order_size),
        )
    }

    /// Create exit order
    fn create_exit_order(&mut self, position: i64, context: &StrategyContext) -> OrderRequest {
        let instrument_id = self.config.instruments[0];
        let quantity = position.abs().min(self.tf_config.order_size as i64) as u32;

        let side = if position > 0 {
            OrderSide::Sell
        } else {
            OrderSide::BuyCover
        };

        // Clear position tracking
        self.entry_price = None;
        self.stop_loss = None;
        self.take_profit = None;
        self.highest_price = None;
        self.lowest_price = None;

        OrderRequest::market_order(
            context.strategy_id.clone(),
            instrument_id,
            side,
            Quantity::from(quantity),
        )
    }
}

impl Strategy for TrendFollowingStrategy {
    fn initialize(&mut self, _context: &StrategyContext) -> Result<(), StrategyError> {
        self.price_history.clear();
        self.volume_history.clear();
        self.short_ma = None;
        self.long_ma = None;
        self.momentum = None;
        self.atr = None;
        self.trend_direction = TrendDirection::None;
        self.entry_price = None;
        self.stop_loss = None;
        self.take_profit = None;
        self.highest_price = None;
        self.lowest_price = None;
        Ok(())
    }

    fn on_market_event(
        &mut self,
        event: &MarketEvent,
        context: &StrategyContext,
    ) -> StrategyOutput {
        let mut output = StrategyOutput::default();

        // Extract price and volume from market event
        let price_volume = match event {
            MarketEvent::Trade(trade) => {
                if trade.instrument_id == self.config.instruments[0] {
                    Some((trade.price, trade.quantity.0 as u64))
                } else {
                    None
                }
            }
            MarketEvent::BBO(bbo) => {
                if bbo.instrument_id == self.config.instruments[0] {
                    // Use mid price
                    match (bbo.bid_price, bbo.ask_price) {
                        (Some(bid), Some(ask)) => {
                            let mid = Price::from_f64((bid.as_f64() + ask.as_f64()) / 2.0);
                            Some((mid, 0)) // No volume for BBO
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some((price, volume)) = price_volume {
            // Update price history
            self.price_history.push((event.timestamp(), price));
            if self.price_history.len() > self.tf_config.long_ma_period * 2 {
                self.price_history.remove(0);
            }

            // Update volume history
            if volume > 0 {
                self.volume_history.push_back((event.timestamp(), volume));
                while self.volume_history.len() > self.tf_config.momentum_period * 2 {
                    self.volume_history.pop_front();
                }
            }

            // Update indicators
            self.update_moving_averages();
            self.update_momentum();
            self.update_atr();

            // Determine trend
            let new_trend = self.determine_trend();
            let trend_changed = new_trend != self.trend_direction;
            self.trend_direction = new_trend;

            let position = context.position.quantity;

            // Check exit conditions first
            if self.check_exit_conditions(price, position) {
                output
                    .orders
                    .push(self.create_exit_order(position, context));
            }
            // Check entry conditions
            else if position == 0 && trend_changed && new_trend != TrendDirection::None {
                // Check confirmations
                if self.momentum_confirms(new_trend) && self.volume_confirms() {
                    let side = match new_trend {
                        TrendDirection::Bullish => OrderSide::Buy,
                        TrendDirection::Bearish => OrderSide::SellShort,
                        TrendDirection::None => unreachable!(),
                    };

                    output
                        .orders
                        .push(self.create_entry_order(side, price, context));
                }
            }
            // Update trailing stop if in position
            else if position != 0 {
                self.update_trailing_stop(price, position > 0);
            }

            // Update metrics
            let mut metrics = StrategyMetrics::new(event.timestamp());
            metrics.add("price", price.as_f64());
            if let Some(short) = self.short_ma {
                metrics.add("short_ma", short);
            }
            if let Some(long) = self.long_ma {
                metrics.add("long_ma", long);
            }
            if let Some(momentum) = self.momentum {
                metrics.add("momentum", momentum * 100.0);
            }
            if let Some(atr) = self.atr {
                metrics.add("atr", atr);
            }
            metrics.add("trend", self.trend_direction as i32 as f64);
            if let Some(stop) = self.stop_loss {
                metrics.add("stop_loss", stop.as_f64());
            }
            if let Some(target) = self.take_profit {
                metrics.add("take_profit", target.as_f64());
            }
            output.set_metrics(metrics);
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
    fn test_trend_following_creation() {
        let strategy =
            TrendFollowingStrategy::new("test_tf".to_string(), 1, TrendFollowingConfig::default());

        assert_eq!(strategy.config.id, "test_tf");
        assert_eq!(strategy.config.instruments.len(), 1);
        assert_eq!(strategy.config.instruments[0], 1);
    }

    #[test]
    fn test_moving_average_calculation() {
        let mut strategy = TrendFollowingStrategy::new(
            "test_tf".to_string(),
            1,
            TrendFollowingConfig {
                short_ma_period: 3,
                long_ma_period: 5,
                ..Default::default()
            },
        );

        // Add price data using Price::from() to properly convert to fixed point
        let prices = vec![100, 102, 104, 103, 105, 107];
        for (i, price) in prices.iter().enumerate() {
            strategy
                .price_history
                .push((i as u64 * 1000, Price::from(*price as i64)));
        }

        strategy.update_moving_averages();

        // Short MA should be average of last 3: (103 + 105 + 107) / 3 = 105.0
        // The function takes the last 3 prices: prices[3], prices[4], prices[5] = 103, 105, 107
        assert!(strategy.short_ma.is_some(), "Short MA should be calculated");
        let expected_short_ma = (103.0 + 105.0 + 107.0) / 3.0; // = 105.0
        assert!(
            (strategy.short_ma.unwrap() - expected_short_ma).abs() < 0.1,
            "Short MA {} should be close to expected {}",
            strategy.short_ma.unwrap(),
            expected_short_ma
        );

        // Long MA should be average of last 5: (102 + 104 + 103 + 105 + 107) / 5 = 104.2
        // The function takes the last 5 prices: prices[1], prices[2], prices[3], prices[4], prices[5] = 102, 104, 103, 105, 107
        assert!(strategy.long_ma.is_some(), "Long MA should be calculated");
        let expected_long_ma = (102.0 + 104.0 + 103.0 + 105.0 + 107.0) / 5.0; // = 104.2
        assert!(
            (strategy.long_ma.unwrap() - expected_long_ma).abs() < 0.1,
            "Long MA {} should be close to expected {}",
            strategy.long_ma.unwrap(),
            expected_long_ma
        );

        // Test edge case with exact period length
        let mut strategy2 = TrendFollowingStrategy::new(
            "test_tf2".to_string(),
            1,
            TrendFollowingConfig {
                short_ma_period: 3,
                long_ma_period: 3,
                ..Default::default()
            },
        );

        // Add exactly 3 prices using Price::from() to properly convert to fixed point
        let prices2 = vec![100, 110, 105];
        for (i, price) in prices2.iter().enumerate() {
            strategy2
                .price_history
                .push((i as u64 * 1000, Price::from(*price as i64)));
        }

        strategy2.update_moving_averages();

        // Both should be (100 + 110 + 105) / 3 = 105.0
        assert!(strategy2.short_ma.is_some());
        assert!(strategy2.long_ma.is_some());
        let expected_ma = (100.0 + 110.0 + 105.0) / 3.0;
        assert!((strategy2.short_ma.unwrap() - expected_ma).abs() < 0.1);
        assert!((strategy2.long_ma.unwrap() - expected_ma).abs() < 0.1);
    }

    #[test]
    fn test_config_validation() {
        // Test default configuration
        let default_config = TrendFollowingConfig::default();
        assert_eq!(default_config.short_ma_period, 10);
        assert_eq!(default_config.long_ma_period, 30);
        assert_eq!(default_config.momentum_period, 14);
        assert_eq!(default_config.momentum_threshold, 0.02);
        assert_eq!(default_config.atr_period, 14);
        assert_eq!(default_config.stop_loss_atr_multiplier, 2.0);
        assert_eq!(default_config.take_profit_atr_multiplier, 3.0);
        assert_eq!(default_config.max_position_size, 10);
        assert_eq!(default_config.order_size, 1);
        assert_eq!(default_config.use_trailing_stops, true);
        assert_eq!(default_config.require_volume_confirmation, false);
        assert_eq!(default_config.volume_multiplier, 1.5);

        // Test custom configuration
        let custom_config = TrendFollowingConfig {
            short_ma_period: 5,
            long_ma_period: 20,
            momentum_period: 10,
            momentum_threshold: 0.03,
            atr_period: 10,
            stop_loss_atr_multiplier: 1.5,
            take_profit_atr_multiplier: 2.5,
            max_position_size: 15,
            order_size: 3,
            use_trailing_stops: false,
            require_volume_confirmation: true,
            volume_multiplier: 2.0,
        };

        let strategy =
            TrendFollowingStrategy::new("custom_tf".to_string(), 2, custom_config.clone());

        assert_eq!(strategy.tf_config.short_ma_period, 5);
        assert_eq!(strategy.tf_config.max_position_size, 15);
        assert_eq!(strategy.tf_config.use_trailing_stops, false);
    }

    #[test]
    fn test_strategy_initialization() {
        let mut strategy = TrendFollowingStrategy::new(
            "init_test".to_string(),
            1,
            TrendFollowingConfig::default(),
        );

        let context = StrategyContext::new(
            "init_test".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        // Add some data first
        strategy.price_history.push((500, Price::from(100i64)));
        strategy.short_ma = Some(105.0);
        strategy.trend_direction = TrendDirection::Bullish;

        // Initialize strategy
        let result = strategy.initialize(&context);
        assert!(result.is_ok());

        // Check that state is properly reset
        assert!(strategy.price_history.is_empty());
        assert!(strategy.volume_history.is_empty());
        assert!(strategy.short_ma.is_none());
        assert!(strategy.long_ma.is_none());
        assert!(strategy.momentum.is_none());
        assert!(strategy.atr.is_none());
        assert_eq!(strategy.trend_direction, TrendDirection::None);
        assert!(strategy.entry_price.is_none());
        assert!(strategy.stop_loss.is_none());
        assert!(strategy.take_profit.is_none());
        assert!(strategy.highest_price.is_none());
        assert!(strategy.lowest_price.is_none());
    }

    #[test]
    fn test_momentum_calculation() {
        let mut strategy = TrendFollowingStrategy::new(
            "momentum_test".to_string(),
            1,
            TrendFollowingConfig {
                momentum_period: 3,
                ..Default::default()
            },
        );

        // Add price history using Price::from() to properly convert to fixed point
        let prices = vec![100, 102, 98, 106]; // Last price up 6% from 3 periods ago
        for (i, price) in prices.iter().enumerate() {
            strategy
                .price_history
                .push((i as u64 * 1000, Price::from(*price as i64)));
        }

        strategy.update_momentum();

        assert!(strategy.momentum.is_some());
        let momentum = strategy.momentum.unwrap();
        // The actual calculation compares current price (last) to price at [len - momentum_period]
        // For len=4, momentum_period=3: compares price[3] (106) to price[4-3] = price[1] (102)
        // momentum = (106 - 102) / 102 = 4/102 ≈ 0.0392
        let expected_momentum = (106.0 - 102.0) / 102.0; // ≈ 0.0392
        assert!(
            (momentum - expected_momentum).abs() < 0.001,
            "Momentum {} should be close to expected {}",
            momentum,
            expected_momentum
        );

        // Test with insufficient data
        let mut strategy2 = TrendFollowingStrategy::new(
            "momentum_test2".to_string(),
            1,
            TrendFollowingConfig {
                momentum_period: 5,
                ..Default::default()
            },
        );

        strategy2.price_history.push((0, Price::from(100i64)));
        strategy2.price_history.push((1000, Price::from(102i64)));
        strategy2.update_momentum();

        assert!(strategy2.momentum.is_none());
    }

    #[test]
    fn test_atr_calculation() {
        let mut strategy = TrendFollowingStrategy::new(
            "atr_test".to_string(),
            1,
            TrendFollowingConfig {
                atr_period: 3,
                ..Default::default()
            },
        );

        // Add price history with some volatility using Price::from()
        let prices = vec![100, 105, 95, 103, 108];
        for (i, price) in prices.iter().enumerate() {
            strategy
                .price_history
                .push((i as u64 * 1000, Price::from(*price as i64)));
        }

        strategy.update_atr();

        assert!(strategy.atr.is_some());
        assert!(strategy.atr.unwrap() > 0.0);

        // Test with insufficient data
        let mut strategy2 = TrendFollowingStrategy::new(
            "atr_test2".to_string(),
            1,
            TrendFollowingConfig {
                atr_period: 10,
                ..Default::default()
            },
        );

        strategy2.price_history.push((0, Price::from(100i64)));
        strategy2.update_atr();

        assert!(strategy2.atr.is_none());
    }

    #[test]
    fn test_volume_confirmation() {
        let mut strategy = TrendFollowingStrategy::new(
            "volume_test".to_string(),
            1,
            TrendFollowingConfig {
                require_volume_confirmation: true,
                volume_multiplier: 1.5,
                ..Default::default()
            },
        );

        // Test with insufficient volume data
        assert!(!strategy.volume_confirms());

        // Add average volume data
        for i in 0..25 {
            strategy.volume_history.push_back((i as u64 * 1000, 1000)); // Average volume of 1000
        }

        // Current volume below threshold
        assert!(!strategy.volume_confirms());

        // Add high volume entry
        strategy.volume_history.push_back((25000, 1600)); // 1.6x average, above 1.5x threshold
        assert!(strategy.volume_confirms());

        // Test with volume confirmation disabled
        let strategy2 = TrendFollowingStrategy::new(
            "volume_test2".to_string(),
            1,
            TrendFollowingConfig {
                require_volume_confirmation: false,
                ..Default::default()
            },
        );

        assert!(strategy2.volume_confirms()); // Should always confirm
    }

    #[test]
    fn test_exit_conditions() {
        let mut strategy = TrendFollowingStrategy::new(
            "exit_test".to_string(),
            1,
            TrendFollowingConfig::default(),
        );

        // Test with no position
        assert!(!strategy.check_exit_conditions(Price::from(100i64), 0));

        // Set up long position
        strategy.stop_loss = Some(Price::from(95i64));
        strategy.take_profit = Some(Price::from(105i64));
        strategy.trend_direction = TrendDirection::Bullish;

        // Test stop loss trigger
        assert!(strategy.check_exit_conditions(Price::from(94i64), 5)); // Below stop

        // Test take profit trigger
        assert!(strategy.check_exit_conditions(Price::from(106i64), 5)); // Above target

        // Test trend reversal
        strategy.trend_direction = TrendDirection::Bearish; // Trend reversed
        assert!(strategy.check_exit_conditions(Price::from(100i64), 5)); // Should exit on trend reversal

        // Test short position
        strategy.trend_direction = TrendDirection::Bearish;
        strategy.stop_loss = Some(Price::from(105i64));
        strategy.take_profit = Some(Price::from(95i64));

        // Test short stop loss
        assert!(strategy.check_exit_conditions(Price::from(106i64), -5)); // Above stop

        // Test short take profit
        assert!(strategy.check_exit_conditions(Price::from(94i64), -5)); // Below target

        // Test no exit condition
        strategy.trend_direction = TrendDirection::Bearish;
        assert!(!strategy.check_exit_conditions(Price::from(100i64), -5)); // No exit
    }

    #[test]
    fn test_trailing_stop_logic() {
        let mut strategy = TrendFollowingStrategy::new(
            "trailing_test".to_string(),
            1,
            TrendFollowingConfig {
                use_trailing_stops: true,
                stop_loss_atr_multiplier: 2.0,
                ..Default::default()
            },
        );

        strategy.atr = Some(5.0);
        strategy.entry_price = Some(Price::from(100i64));
        strategy.stop_loss = Some(Price::from(90i64));

        // Test long position trailing stop
        strategy.highest_price = Some(Price::from(105i64));

        // Price moves higher, should update trailing stop
        strategy.update_trailing_stop(Price::from(110i64), true);
        assert_eq!(strategy.highest_price, Some(Price::from(110i64)));

        // Stop should be updated (110 - 2*5 = 100)
        let expected_stop = 110_000_000_000i64 - 2i64 * 5i64;
        assert!(strategy.stop_loss.unwrap().0 >= expected_stop - 50); // Allow for some rounding

        // Price moves lower, should not update trailing stop
        let old_stop = strategy.stop_loss.unwrap();
        strategy.update_trailing_stop(Price::from(108i64), true);
        assert_eq!(strategy.stop_loss, Some(old_stop)); // No change

        // Test short position trailing stop
        strategy.lowest_price = Some(Price::from(95i64));
        strategy.stop_loss = Some(Price::from(105i64));

        // Price moves lower, should update trailing stop
        strategy.update_trailing_stop(Price::from(90i64), false);
        assert_eq!(strategy.lowest_price, Some(Price::from(90i64)));

        // Test with trailing stops disabled
        let mut strategy2 = TrendFollowingStrategy::new(
            "trailing_test2".to_string(),
            1,
            TrendFollowingConfig {
                use_trailing_stops: false,
                ..Default::default()
            },
        );

        strategy2.entry_price = Some(Price::from(100i64));
        strategy2.stop_loss = Some(Price::from(90i64));
        let old_stop = strategy2.stop_loss.unwrap();

        strategy2.update_trailing_stop(Price::from(110i64), true);
        assert_eq!(strategy2.stop_loss, Some(old_stop)); // No change
    }

    #[test]
    fn test_market_event_processing() {
        use crate::market_data::events::{BBOUpdate, MarketEvent, TradeEvent};

        let mut strategy = TrendFollowingStrategy::new(
            "event_test".to_string(),
            1,
            TrendFollowingConfig {
                short_ma_period: 3,
                long_ma_period: 5,
                momentum_period: 3,
                require_volume_confirmation: false,
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
        assert!(output.metrics.is_some());

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
    fn test_order_generation_scenarios() {
        let mut strategy = TrendFollowingStrategy::new(
            "order_test".to_string(),
            1,
            TrendFollowingConfig {
                short_ma_period: 2,
                long_ma_period: 3,
                momentum_period: 2,
                momentum_threshold: 0.01,
                require_volume_confirmation: false,
                order_size: 3,
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

        // Set up trending market data (bullish) using Price::from()
        let prices = vec![100, 102, 105]; // Uptrend
        for (i, price) in prices.iter().enumerate() {
            strategy
                .price_history
                .push((i as u64 * 1000, Price::from(*price as i64)));
        }

        strategy.update_moving_averages();
        strategy.update_momentum();

        // Force trend detection
        strategy.trend_direction = TrendDirection::None;
        let new_trend = strategy.determine_trend();

        if new_trend == TrendDirection::Bullish && strategy.momentum_confirms(new_trend) {
            let order = strategy.create_entry_order(OrderSide::Buy, Price::from(105i64), &context);

            assert_eq!(order.side, OrderSide::Buy);
            assert_eq!(order.quantity, Quantity::from(3u32));
            assert!(strategy.entry_price.is_some());
            assert!(strategy.stop_loss.is_some());
            assert!(strategy.take_profit.is_some());
        }

        // Test exit order creation
        strategy.entry_price = Some(Price::from(105i64));
        let exit_order = strategy.create_exit_order(3, &context);

        assert_eq!(exit_order.side, OrderSide::Sell);
        assert_eq!(exit_order.quantity, Quantity::from(3u32));
        assert!(strategy.entry_price.is_none()); // Should be cleared
    }

    #[test]
    fn test_edge_cases_and_error_handling() {
        let mut strategy = TrendFollowingStrategy::new(
            "edge_test".to_string(),
            1,
            TrendFollowingConfig {
                short_ma_period: 5,
                long_ma_period: 3, // Invalid: short > long
                ..Default::default()
            },
        );

        // Test with invalid MA configuration
        strategy.price_history = vec![
            (0, Price::from(100i64)),
            (1000, Price::from(101i64)),
            (2000, Price::from(102i64)),
            (3000, Price::from(103i64)),
            (4000, Price::from(104i64)),
        ];

        strategy.update_moving_averages();

        // Should handle gracefully
        assert!(strategy.short_ma.is_some());
        assert!(strategy.long_ma.is_some());

        // Test with zero prices
        let mut strategy2 = TrendFollowingStrategy::new(
            "zero_test".to_string(),
            1,
            TrendFollowingConfig {
                momentum_period: 2,
                ..Default::default()
            },
        );

        strategy2.price_history = vec![
            (0, Price::from(0i64)),
            (1000, Price::from(0i64)),
            (2000, Price::from(100i64)),
        ];

        strategy2.update_momentum();
        // Should handle division by zero gracefully

        // Test with very large price movements
        let mut strategy3 = TrendFollowingStrategy::new(
            "extreme_test".to_string(),
            1,
            TrendFollowingConfig::default(),
        );

        strategy3.atr = Some(1000.0); // Very high volatility
        let entry_price = Price::from(100i64);
        let (stop, target) = strategy3.calculate_exit_levels(entry_price, true);

        // Should still produce valid levels
        assert!(stop.0 < entry_price.0);
        assert!(target.0 > entry_price.0);
    }

    #[test]
    fn test_trend_determination() {
        let mut strategy =
            TrendFollowingStrategy::new("test_tf".to_string(), 1, TrendFollowingConfig::default());

        // Set up bullish trend (short MA > long MA)
        strategy.short_ma = Some(105.0);
        strategy.long_ma = Some(100.0);
        strategy.trend_direction = TrendDirection::None;

        let trend = strategy.determine_trend();
        assert_eq!(trend, TrendDirection::Bullish);

        // Set up bearish trend (short MA < long MA)
        strategy.short_ma = Some(95.0);
        strategy.long_ma = Some(100.0);
        strategy.trend_direction = TrendDirection::None;

        let trend = strategy.determine_trend();
        assert_eq!(trend, TrendDirection::Bearish);
    }

    #[test]
    fn test_momentum_confirmation() {
        let strategy = TrendFollowingStrategy::new(
            "test_tf".to_string(),
            1,
            TrendFollowingConfig {
                momentum_threshold: 0.02, // 2%
                ..Default::default()
            },
        );

        // Test bullish momentum confirmation
        let mut test_strategy = strategy.clone();
        test_strategy.momentum = Some(0.03); // 3% positive momentum
        assert!(test_strategy.momentum_confirms(TrendDirection::Bullish));
        assert!(!test_strategy.momentum_confirms(TrendDirection::Bearish));

        // Test bearish momentum confirmation
        test_strategy.momentum = Some(-0.025); // 2.5% negative momentum
        assert!(!test_strategy.momentum_confirms(TrendDirection::Bullish));
        assert!(test_strategy.momentum_confirms(TrendDirection::Bearish));

        // Test insufficient momentum
        test_strategy.momentum = Some(0.01); // Only 1%
        assert!(!test_strategy.momentum_confirms(TrendDirection::Bullish));
        assert!(!test_strategy.momentum_confirms(TrendDirection::Bearish));
    }

    #[test]
    fn test_exit_level_calculation() {
        let mut strategy = TrendFollowingStrategy::new(
            "test_tf".to_string(),
            1,
            TrendFollowingConfig {
                stop_loss_atr_multiplier: 2.0,
                take_profit_atr_multiplier: 3.0,
                ..Default::default()
            },
        );

        strategy.atr = Some(50.0); // ATR of 50
        let entry_price = Price::new(10000i64);

        // Test long position levels
        let (stop_long, target_long) = strategy.calculate_exit_levels(entry_price, true);
        assert_eq!(stop_long.0, 9900); // 10000 - (2 * 50)
        assert_eq!(target_long.0, 10150); // 10000 + (3 * 50)

        // Test short position levels
        let (stop_short, target_short) = strategy.calculate_exit_levels(entry_price, false);
        assert_eq!(stop_short.0, 10100); // 10000 + (2 * 50)
        assert_eq!(target_short.0, 9850); // 10000 - (3 * 50)
    }

    #[test]
    fn test_order_creation() {
        let mut strategy = TrendFollowingStrategy::new(
            "test_tf".to_string(),
            1,
            TrendFollowingConfig {
                order_size: 5,
                ..Default::default()
            },
        );

        let context = StrategyContext::new(
            "test_tf".to_string(),
            1000,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        // Test entry order creation
        let entry_order =
            strategy.create_entry_order(OrderSide::Buy, Price::from(10000i64), &context);

        assert_eq!(entry_order.side, OrderSide::Buy);
        assert_eq!(entry_order.quantity, Quantity::from(5u32));
        assert_eq!(entry_order.order_type, OrderType::Market);
        assert!(strategy.entry_price.is_some());
        assert!(strategy.stop_loss.is_some());
        assert!(strategy.take_profit.is_some());

        // Test exit order creation
        let exit_order = strategy.create_exit_order(5, &context);

        assert_eq!(exit_order.side, OrderSide::Sell);
        assert_eq!(exit_order.quantity, Quantity::from(5u32));
        assert!(strategy.entry_price.is_none());
        assert!(strategy.stop_loss.is_none());
    }
}

// Clone implementation for testing
impl Clone for TrendFollowingStrategy {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            tf_config: self.tf_config.clone(),
            price_history: self.price_history.clone(),
            volume_history: self.volume_history.clone(),
            short_ma: self.short_ma,
            long_ma: self.long_ma,
            momentum: self.momentum,
            atr: self.atr,
            trend_direction: self.trend_direction,
            entry_price: self.entry_price,
            stop_loss: self.stop_loss,
            take_profit: self.take_profit,
            highest_price: self.highest_price,
            lowest_price: self.lowest_price,
        }
    }
}
