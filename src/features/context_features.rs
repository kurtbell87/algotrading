//! Context-aware features for trading strategies
//!
//! This module provides features that depend on the current strategy state:
//! - Position and inventory features
//! - P&L tracking
//! - Risk exposure metrics
//! - Order state features

use crate::core::types::Price;
use crate::features::collector::FeatureVector;
use std::collections::VecDeque;

/// Position state for feature calculation
#[derive(Debug, Clone)]
pub struct Position {
    /// Current position quantity (positive = long, negative = short)
    pub quantity: i64,
    /// Average entry price
    pub avg_price: Price,
    /// Realized P&L
    pub realized_pnl: f64,
    /// Maximum position size reached
    pub max_position: i64,
    /// Minimum position size reached
    pub min_position: i64,
    /// Number of trades executed
    pub trade_count: u64,
    /// Total volume traded
    pub total_volume: u64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            quantity: 0,
            avg_price: Price::new(0),
            realized_pnl: 0.0,
            max_position: 0,
            min_position: 0,
            trade_count: 0,
            total_volume: 0,
        }
    }
}

/// Risk limits for position management
#[derive(Debug, Clone)]
pub struct RiskLimits {
    /// Maximum allowed position size
    pub max_position: i64,
    /// Maximum order size
    pub max_order_size: i64,
    /// Maximum loss allowed
    pub max_loss: f64,
    /// Maximum daily loss
    pub daily_max_loss: f64,
    /// Maximum orders per minute
    pub max_orders_per_minute: u32,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position: 100,
            max_order_size: 50,
            max_loss: 10000.0,
            daily_max_loss: 5000.0,
            max_orders_per_minute: 100,
        }
    }
}

/// P&L tracking over time
#[derive(Debug, Clone)]
struct PnLHistory {
    /// Rolling P&L values (timestamp, pnl)
    history: VecDeque<(u64, f64)>,
    /// Maximum P&L reached
    high_water_mark: f64,
    /// Current drawdown from high water mark
    current_drawdown: f64,
    /// Maximum drawdown experienced
    max_drawdown: f64,
    /// Daily P&L
    daily_pnl: f64,
    /// Last reset timestamp (for daily calculations)
    last_reset: u64,
}

impl Default for PnLHistory {
    fn default() -> Self {
        Self {
            history: VecDeque::with_capacity(1000),
            high_water_mark: 0.0,
            current_drawdown: 0.0,
            max_drawdown: 0.0,
            daily_pnl: 0.0,
            last_reset: 0,
        }
    }
}

/// Order activity tracking
#[derive(Debug, Clone)]
struct OrderActivity {
    /// Recent order timestamps
    order_timestamps: VecDeque<u64>,
    /// Orders by type in current minute
    market_orders: u32,
    limit_orders: u32,
    cancel_count: u32,
    /// Fill rate tracking
    orders_sent: u32,
    orders_filled: u32,
}

impl Default for OrderActivity {
    fn default() -> Self {
        Self {
            order_timestamps: VecDeque::with_capacity(100),
            market_orders: 0,
            limit_orders: 0,
            cancel_count: 0,
            orders_sent: 0,
            orders_filled: 0,
        }
    }
}

/// Context-aware features extractor
pub struct ContextFeatures {
    /// Current position
    pub position: Position,
    /// Risk limits
    risk_limits: RiskLimits,
    /// P&L history
    pnl_history: PnLHistory,
    /// Order activity
    order_activity: OrderActivity,
    /// Target position (for inventory management)
    target_position: i64,
    /// Position hold time tracking
    position_entry_time: Option<u64>,
    /// Consecutive wins/losses
    consecutive_wins: u32,
    consecutive_losses: u32,
}

impl ContextFeatures {
    pub fn new(risk_limits: RiskLimits) -> Self {
        Self {
            position: Position::default(),
            risk_limits,
            pnl_history: PnLHistory::default(),
            order_activity: OrderActivity::default(),
            target_position: 0,
            position_entry_time: None,
            consecutive_wins: 0,
            consecutive_losses: 0,
        }
    }

    /// Update position from fill
    pub fn update_position(&mut self, fill_price: Price, fill_quantity: i64, timestamp: u64) {
        let old_quantity = self.position.quantity;
        self.position.quantity += fill_quantity;
        self.position.trade_count += 1;
        self.position.total_volume += fill_quantity.unsigned_abs();

        // Update position tracking
        self.position.max_position = self.position.max_position.max(self.position.quantity);
        self.position.min_position = self.position.min_position.min(self.position.quantity);

        // Update average price
        if old_quantity == 0 {
            // New position
            self.position.avg_price = fill_price;
            self.position_entry_time = Some(timestamp);
        } else if old_quantity.signum() == fill_quantity.signum() {
            // Adding to position
            let total_cost = self.position.avg_price.as_f64() * old_quantity.abs() as f64
                + fill_price.as_f64() * fill_quantity.abs() as f64;
            let total_quantity = old_quantity.abs() + fill_quantity.abs();
            self.position.avg_price = Price::from_f64(total_cost / total_quantity as f64);
        } else {
            // Reducing or flipping position
            let closed_quantity = old_quantity.abs().min(fill_quantity.abs());
            let pnl = closed_quantity as f64
                * (fill_price.as_f64() - self.position.avg_price.as_f64())
                * old_quantity.signum() as f64;

            self.position.realized_pnl += pnl;
            self.update_pnl_tracking(pnl, timestamp);

            // Update consecutive tracking
            if pnl > 0.0 {
                self.consecutive_wins += 1;
                self.consecutive_losses = 0;
            } else if pnl < 0.0 {
                self.consecutive_losses += 1;
                self.consecutive_wins = 0;
            }

            // If flipped, update avg price
            if self.position.quantity != 0
                && old_quantity.signum() != self.position.quantity.signum()
            {
                self.position.avg_price = fill_price;
                self.position_entry_time = Some(timestamp);
            } else if self.position.quantity == 0 {
                self.position_entry_time = None;
            }
        }
    }

    /// Update P&L tracking
    fn update_pnl_tracking(&mut self, pnl_change: f64, timestamp: u64) {
        let total_pnl = self.position.realized_pnl;

        // Update history
        self.pnl_history.history.push_back((timestamp, total_pnl));
        if self.pnl_history.history.len() > 1000 {
            self.pnl_history.history.pop_front();
        }

        // Update high water mark and drawdown
        if total_pnl > self.pnl_history.high_water_mark {
            self.pnl_history.high_water_mark = total_pnl;
            self.pnl_history.current_drawdown = 0.0;
        } else {
            self.pnl_history.current_drawdown = self.pnl_history.high_water_mark - total_pnl;
            self.pnl_history.max_drawdown = self
                .pnl_history
                .max_drawdown
                .max(self.pnl_history.current_drawdown);
        }

        // Update daily P&L (reset at midnight UTC)
        let day = timestamp / (24 * 3600 * 1_000_000);
        let last_day = self.pnl_history.last_reset / (24 * 3600 * 1_000_000);

        if day != last_day {
            self.pnl_history.daily_pnl = pnl_change;
            self.pnl_history.last_reset = timestamp;
        } else {
            self.pnl_history.daily_pnl += pnl_change;
        }
    }

    /// Update order activity
    pub fn update_order_sent(&mut self, is_market: bool, timestamp: u64) {
        self.order_activity.order_timestamps.push_back(timestamp);

        // Clean old timestamps (older than 1 minute)
        let cutoff = timestamp.saturating_sub(60_000_000); // 60 seconds
        while let Some(&front_time) = self.order_activity.order_timestamps.front() {
            if front_time < cutoff {
                self.order_activity.order_timestamps.pop_front();
            } else {
                break;
            }
        }

        if is_market {
            self.order_activity.market_orders += 1;
        } else {
            self.order_activity.limit_orders += 1;
        }
        self.order_activity.orders_sent += 1;
    }

    /// Update order filled
    pub fn update_order_filled(&mut self) {
        self.order_activity.orders_filled += 1;
    }

    /// Update order cancelled
    pub fn update_order_cancelled(&mut self) {
        self.order_activity.cancel_count += 1;
    }

    /// Set target position for inventory management
    pub fn set_target_position(&mut self, target: i64) {
        self.target_position = target;
    }

    /// Calculate unrealized P&L
    pub fn unrealized_pnl(&self, current_price: Price) -> f64 {
        if self.position.quantity == 0 {
            0.0
        } else {
            self.position.quantity as f64
                * (current_price.as_f64() - self.position.avg_price.as_f64())
        }
    }

    /// Calculate total P&L
    pub fn total_pnl(&self, current_price: Price) -> f64 {
        self.position.realized_pnl + self.unrealized_pnl(current_price)
    }

    /// Calculate position hold time in seconds
    fn position_hold_time(&self, timestamp: u64) -> Option<f64> {
        self.position_entry_time.map(|entry| {
            (timestamp - entry) as f64 / 1_000_000.0 // Convert to seconds
        })
    }

    /// Calculate fill rate
    fn fill_rate(&self) -> f64 {
        if self.order_activity.orders_sent > 0 {
            self.order_activity.orders_filled as f64 / self.order_activity.orders_sent as f64
        } else {
            0.0
        }
    }

    /// Add context features to feature vector
    pub fn add_to_vector(
        &self,
        features: &mut FeatureVector,
        current_price: Price,
        timestamp: u64,
    ) {
        // Position features
        features.add("position_size", self.position.quantity as f64);
        features.add("position_abs_size", self.position.quantity.abs() as f64);
        features.add("position_side", self.position.quantity.signum() as f64);
        features.add(
            "is_flat",
            if self.position.quantity == 0 {
                1.0
            } else {
                0.0
            },
        );
        features.add(
            "is_long",
            if self.position.quantity > 0 { 1.0 } else { 0.0 },
        );
        features.add(
            "is_short",
            if self.position.quantity < 0 { 1.0 } else { 0.0 },
        );

        // Inventory management
        let inventory_deviation = (self.position.quantity - self.target_position) as f64;
        features.add("inventory_deviation", inventory_deviation);
        features.add(
            "inventory_deviation_pct",
            inventory_deviation / self.risk_limits.max_position as f64,
        );

        // Position utilization
        features.add(
            "position_utilization",
            self.position.quantity.abs() as f64 / self.risk_limits.max_position as f64,
        );
        features.add("max_position_reached", self.position.max_position as f64);
        features.add("min_position_reached", self.position.min_position as f64);

        // P&L features
        let unrealized = self.unrealized_pnl(current_price);
        let total_pnl = self.total_pnl(current_price);

        features.add("realized_pnl", self.position.realized_pnl);
        features.add("unrealized_pnl", unrealized);
        features.add("total_pnl", total_pnl);
        features.add("daily_pnl", self.pnl_history.daily_pnl);

        // Normalized P&L
        features.add(
            "pnl_per_contract",
            if self.position.total_volume > 0 {
                self.position.realized_pnl / self.position.total_volume as f64
            } else {
                0.0
            },
        );

        // Drawdown features
        features.add("current_drawdown", self.pnl_history.current_drawdown);
        features.add("max_drawdown", self.pnl_history.max_drawdown);
        features.add(
            "drawdown_pct",
            if self.pnl_history.high_water_mark > 0.0 {
                self.pnl_history.current_drawdown / self.pnl_history.high_water_mark
            } else {
                0.0
            },
        );

        // Risk utilization
        features.add(
            "loss_utilization",
            (-total_pnl).max(0.0) / self.risk_limits.max_loss,
        );
        features.add(
            "daily_loss_utilization",
            (-self.pnl_history.daily_pnl).max(0.0) / self.risk_limits.daily_max_loss,
        );

        // Trading activity
        features.add("trade_count", self.position.trade_count as f64);
        features.add("total_volume", self.position.total_volume as f64);
        features.add(
            "avg_trade_size",
            if self.position.trade_count > 0 {
                self.position.total_volume as f64 / self.position.trade_count as f64
            } else {
                0.0
            },
        );

        // Order activity
        features.add(
            "orders_per_minute",
            self.order_activity.order_timestamps.len() as f64,
        );
        features.add(
            "order_rate_utilization",
            self.order_activity.order_timestamps.len() as f64
                / self.risk_limits.max_orders_per_minute as f64,
        );
        features.add(
            "market_orders_count",
            self.order_activity.market_orders as f64,
        );
        features.add(
            "limit_orders_count",
            self.order_activity.limit_orders as f64,
        );
        features.add(
            "cancel_rate",
            if self.order_activity.orders_sent > 0 {
                self.order_activity.cancel_count as f64 / self.order_activity.orders_sent as f64
            } else {
                0.0
            },
        );
        features.add("fill_rate", self.fill_rate());

        // Position timing
        if let Some(hold_time) = self.position_hold_time(timestamp) {
            features.add("position_hold_seconds", hold_time);
            features.add("position_hold_minutes", hold_time / 60.0);
        }

        // Streak features
        features.add("consecutive_wins", self.consecutive_wins as f64);
        features.add("consecutive_losses", self.consecutive_losses as f64);
        features.add(
            "on_winning_streak",
            if self.consecutive_wins > 0 { 1.0 } else { 0.0 },
        );
        features.add(
            "on_losing_streak",
            if self.consecutive_losses > 0 {
                1.0
            } else {
                0.0
            },
        );

        // Entry price features
        if self.position.quantity != 0 {
            let price_from_entry = current_price.as_f64() - self.position.avg_price.as_f64();
            features.add("price_from_entry", price_from_entry);
            features.add(
                "price_from_entry_pct",
                price_from_entry / self.position.avg_price.as_f64() * 100.0,
            );
            features.add("entry_price", self.position.avg_price.as_f64());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_update_new_position() {
        let mut context = ContextFeatures::new(RiskLimits::default());

        // Open long position
        context.update_position(Price::from(100i64), 10, 1000);

        assert_eq!(context.position.quantity, 10);
        assert_eq!(context.position.avg_price, Price::from(100i64));
        assert_eq!(context.position.trade_count, 1);
        assert_eq!(context.position.total_volume, 10);
    }

    #[test]
    fn test_position_add_to_position() {
        let mut context = ContextFeatures::new(RiskLimits::default());

        // Open position
        context.update_position(Price::from(100i64), 10, 1000);

        // Add to position
        context.update_position(Price::from(102i64), 5, 2000);

        assert_eq!(context.position.quantity, 15);
        // Average price = (100*10 + 102*5) / 15 = 1510/15 = 100.67
        let expected_avg = Price::from_f64(100.66666666666667);
        assert_eq!(context.position.avg_price, expected_avg);
    }

    #[test]
    fn test_position_close_with_profit() {
        let mut context = ContextFeatures::new(RiskLimits::default());

        // Open long
        context.update_position(Price::from(100i64), 10, 1000);

        // Close with profit
        context.update_position(Price::from(105i64), -10, 2000);

        assert_eq!(context.position.quantity, 0);
        assert_eq!(context.position.realized_pnl, 50.0); // 10 * (105 - 100)
        assert_eq!(context.consecutive_wins, 1);
        assert_eq!(context.consecutive_losses, 0);
    }

    #[test]
    fn test_position_flip() {
        let mut context = ContextFeatures::new(RiskLimits::default());

        // Open long
        context.update_position(Price::from(100i64), 10, 1000);

        // Flip to short
        context.update_position(Price::from(102i64), -15, 2000);

        assert_eq!(context.position.quantity, -5);
        assert_eq!(context.position.avg_price, Price::from(102i64));
        assert_eq!(context.position.realized_pnl, 20.0); // 10 * (102 - 100)
    }

    #[test]
    fn test_pnl_tracking() {
        let mut context = ContextFeatures::new(RiskLimits::default());

        // Make profitable trade
        context.update_position(Price::from(100i64), 10, 1000);
        context.update_position(Price::from(110i64), -10, 2000);

        assert_eq!(context.position.realized_pnl, 100.0);
        assert_eq!(context.pnl_history.high_water_mark, 100.0);
        assert_eq!(context.pnl_history.current_drawdown, 0.0);

        // Make losing trade
        context.update_position(Price::from(110i64), 10, 3000);
        context.update_position(Price::from(105i64), -10, 4000);

        assert_eq!(context.position.realized_pnl, 50.0); // 100 - 50
        assert_eq!(context.pnl_history.high_water_mark, 100.0);
        assert_eq!(context.pnl_history.current_drawdown, 50.0);
        assert_eq!(context.pnl_history.max_drawdown, 50.0);
    }

    #[test]
    fn test_order_activity_tracking() {
        let mut context = ContextFeatures::new(RiskLimits::default());

        // Send some orders
        context.update_order_sent(true, 1_000_000); // Market
        context.update_order_sent(false, 2_000_000); // Limit
        context.update_order_sent(false, 3_000_000); // Limit

        // Fill one
        context.update_order_filled();

        // Cancel one
        context.update_order_cancelled();

        assert_eq!(context.order_activity.market_orders, 1);
        assert_eq!(context.order_activity.limit_orders, 2);
        assert_eq!(context.order_activity.orders_sent, 3);
        assert_eq!(context.order_activity.orders_filled, 1);
        assert_eq!(context.order_activity.cancel_count, 1);
        assert_eq!(context.fill_rate(), 1.0 / 3.0);
    }

    #[test]
    fn test_feature_vector_integration() {
        let mut context = ContextFeatures::new(RiskLimits::default());

        // Set up some state
        context.set_target_position(5);
        context.update_position(Price::from(100i64), 10, 1000);
        context.update_order_sent(false, 1000);
        context.update_order_filled();

        let mut features = FeatureVector::new(1, 2000);
        context.add_to_vector(&mut features, Price::from(102i64), 2000);

        // Check key features
        assert_eq!(features.get("position_size"), Some(10.0));
        assert_eq!(features.get("is_long"), Some(1.0));
        assert_eq!(features.get("inventory_deviation"), Some(5.0)); // 10 - 5
        assert_eq!(features.get("unrealized_pnl"), Some(20.0)); // 10 * (102 - 100)
        assert_eq!(features.get("fill_rate"), Some(1.0));
    }
}
