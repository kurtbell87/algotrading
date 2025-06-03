//! Position management for backtesting
//!
//! This module handles position tracking, P&L calculation, and risk monitoring
//! for individual strategies and portfolios.

use crate::backtest::events::FillEvent;
use crate::core::Side;
use crate::core::types::{InstrumentId, Price};
use crate::features::RiskLimits;
use crate::strategy::StrategyId;
use std::collections::HashMap;

/// A single position in an instrument
#[derive(Debug, Clone)]
pub struct Position {
    /// Instrument identifier
    pub instrument_id: InstrumentId,
    /// Current quantity (positive = long, negative = short)
    pub quantity: i64,
    /// Average entry price
    pub avg_price: Price,
    /// Total cost basis (including commissions)
    pub cost_basis: f64,
    /// Realized P&L from closed trades
    pub realized_pnl: f64,
    /// Unrealized P&L (mark-to-market)
    pub unrealized_pnl: f64,
    /// Current market price for mark-to-market
    pub market_price: Option<Price>,
    /// Timestamp of last update
    pub last_update: u64,
    /// Commission paid on this position
    pub total_commission: f64,
}

impl Position {
    /// Create a new empty position
    pub fn new(instrument_id: InstrumentId) -> Self {
        Self {
            instrument_id,
            quantity: 0,
            avg_price: Price::from(0i64),
            cost_basis: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            market_price: None,
            last_update: 0,
            total_commission: 0.0,
        }
    }

    /// Check if position is flat (no holdings)
    pub fn is_flat(&self) -> bool {
        self.quantity == 0
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.quantity > 0
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.quantity < 0
    }

    /// Get absolute size of position
    pub fn abs_quantity(&self) -> u64 {
        self.quantity.abs() as u64
    }

    /// Apply a fill to this position
    pub fn apply_fill(&mut self, fill: &FillEvent) {
        let fill_quantity = match fill.side {
            Side::Bid => fill.quantity.as_i64(),  // Buy increases position
            Side::Ask => -fill.quantity.as_i64(), // Sell decreases position
        };

        let fill_value = fill.price.as_f64() * fill.quantity.as_f64();
        let fill_cost = fill_value + fill.commission;

        // Check if this closes or reduces existing position
        if (self.quantity > 0 && fill_quantity < 0) || (self.quantity < 0 && fill_quantity > 0) {
            // Closing or reducing position - calculate realized P&L
            let closing_quantity = fill_quantity.abs().min(self.quantity.abs());

            if closing_quantity > 0 {
                let avg_cost_per_unit = self.cost_basis / self.quantity.abs() as f64;
                let realized_cost = avg_cost_per_unit * closing_quantity as f64;

                let closing_value = if self.quantity > 0 {
                    // Closing long position
                    fill_value - fill.commission
                } else {
                    // Closing short position
                    -fill_value - fill.commission
                };

                self.realized_pnl += closing_value - realized_cost;

                // Adjust cost basis for remaining position
                let remaining_quantity = self.quantity.abs() - closing_quantity;
                if remaining_quantity > 0 {
                    self.cost_basis = avg_cost_per_unit * remaining_quantity as f64;
                } else {
                    self.cost_basis = 0.0;
                }
            }
        }

        // Update position quantity
        let new_quantity = self.quantity + fill_quantity;

        if new_quantity == 0 {
            // Position is now flat
            self.quantity = 0;
            self.avg_price = Price::from(0i64);
            self.cost_basis = 0.0;
        } else if (self.quantity > 0 && new_quantity > 0) || (self.quantity < 0 && new_quantity < 0)
        {
            // Adding to existing position - update average price
            let total_cost = self.cost_basis + fill_cost;
            self.quantity = new_quantity;
            self.cost_basis = total_cost;
            self.avg_price = Price::from_f64(total_cost / new_quantity.abs() as f64);
        } else {
            // New position (reversed from previous or starting from flat)
            self.quantity = new_quantity;
            self.cost_basis = fill_cost.abs();
            self.avg_price = fill.price;
        }

        self.total_commission += fill.commission;
        self.last_update = fill.timestamp;
    }

    /// Update market price for mark-to-market calculation
    pub fn update_market_price(&mut self, price: Price, timestamp: u64) {
        self.market_price = Some(price);
        self.last_update = timestamp;

        // Recalculate unrealized P&L
        if self.quantity != 0 {
            let market_value = price.as_f64() * self.quantity.abs() as f64;

            if self.quantity > 0 {
                // Long position
                self.unrealized_pnl = market_value - self.cost_basis;
            } else {
                // Short position
                self.unrealized_pnl = self.cost_basis - market_value;
            }
        } else {
            self.unrealized_pnl = 0.0;
        }
    }

    /// Get total P&L (realized + unrealized)
    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl
    }

    /// Get current market value of position
    pub fn market_value(&self) -> f64 {
        if let Some(price) = self.market_price {
            price.as_f64() * self.quantity.abs() as f64
        } else {
            self.cost_basis
        }
    }

    /// Get return on investment percentage
    pub fn roi_percent(&self) -> f64 {
        if self.cost_basis != 0.0 {
            (self.total_pnl() / self.cost_basis) * 100.0
        } else {
            0.0
        }
    }
}

/// Tracks positions for a single strategy
#[derive(Debug, Clone)]
pub struct PositionTracker {
    /// Strategy identifier
    pub strategy_id: StrategyId,
    /// Positions by instrument
    positions: HashMap<InstrumentId, Position>,
    /// Risk limits
    risk_limits: RiskLimits,
    /// Total realized P&L across all instruments
    total_realized_pnl: f64,
    /// Total unrealized P&L across all instruments
    total_unrealized_pnl: f64,
    /// High water mark for drawdown calculation
    high_water_mark: f64,
    /// Current drawdown from high water mark
    current_drawdown: f64,
    /// Maximum drawdown experienced
    max_drawdown: f64,
    /// Total commission paid
    total_commission: f64,
    /// Daily P&L (resets each day)
    daily_pnl: f64,
    /// Last daily reset timestamp
    last_daily_reset: u64,
}

impl PositionTracker {
    /// Create a new position tracker
    pub fn new(strategy_id: StrategyId, risk_limits: RiskLimits) -> Self {
        Self {
            strategy_id,
            positions: HashMap::new(),
            risk_limits,
            total_realized_pnl: 0.0,
            total_unrealized_pnl: 0.0,
            high_water_mark: 0.0,
            current_drawdown: 0.0,
            max_drawdown: 0.0,
            total_commission: 0.0,
            daily_pnl: 0.0,
            last_daily_reset: 0,
        }
    }

    /// Apply a fill to the appropriate position
    pub fn apply_fill(&mut self, fill: &FillEvent) {
        let position = self
            .positions
            .entry(fill.instrument_id)
            .or_insert_with(|| Position::new(fill.instrument_id));

        let old_realized = position.realized_pnl;
        position.apply_fill(fill);

        // Update totals
        let realized_change = position.realized_pnl - old_realized;
        self.total_realized_pnl += realized_change;
        self.total_commission += fill.commission;
        self.daily_pnl += realized_change;

        // Update high water mark and drawdown
        self.update_drawdown();
    }

    /// Update market prices for all positions
    pub fn update_market_prices(&mut self, prices: &HashMap<InstrumentId, Price>, timestamp: u64) {
        self.total_unrealized_pnl = 0.0;

        for (instrument_id, position) in &mut self.positions {
            if let Some(&price) = prices.get(instrument_id) {
                position.update_market_price(price, timestamp);
                self.total_unrealized_pnl += position.unrealized_pnl;
            }
        }

        // Update drawdown based on total P&L
        self.update_drawdown();
    }

    /// Update drawdown calculation
    fn update_drawdown(&mut self) {
        let total_pnl = self.total_pnl();

        if total_pnl > self.high_water_mark {
            self.high_water_mark = total_pnl;
            self.current_drawdown = 0.0;
        } else {
            self.current_drawdown = self.high_water_mark - total_pnl;
            if self.current_drawdown > self.max_drawdown {
                self.max_drawdown = self.current_drawdown;
            }
        }
    }

    /// Reset daily P&L tracking
    pub fn reset_daily_pnl(&mut self, timestamp: u64) {
        self.daily_pnl = 0.0;
        self.last_daily_reset = timestamp;
    }

    /// Get position for an instrument
    pub fn get_position(&self, instrument_id: InstrumentId) -> Option<&Position> {
        self.positions.get(&instrument_id)
    }

    /// Get all positions
    pub fn get_all_positions(&self) -> &HashMap<InstrumentId, Position> {
        &self.positions
    }

    /// Get total quantity for an instrument
    pub fn get_quantity(&self, instrument_id: InstrumentId) -> i64 {
        self.positions
            .get(&instrument_id)
            .map(|p| p.quantity)
            .unwrap_or(0)
    }

    /// Check if position is within risk limits
    pub fn check_risk_limits(
        &self,
        instrument_id: InstrumentId,
        new_quantity: i64,
    ) -> Result<(), RiskViolation> {
        // Check position size limit
        if new_quantity.abs() > self.risk_limits.max_position {
            return Err(RiskViolation::PositionSizeExceeded {
                instrument_id,
                current: new_quantity.abs() as u64,
                limit: self.risk_limits.max_position as u64,
            });
        }

        // Check total loss limit
        let total_pnl = self.total_pnl();
        if total_pnl < -self.risk_limits.max_loss {
            return Err(RiskViolation::MaxLossExceeded {
                current_loss: -total_pnl,
                limit: self.risk_limits.max_loss,
            });
        }

        // Check daily loss limit
        if self.daily_pnl < -self.risk_limits.daily_max_loss {
            return Err(RiskViolation::DailyLossExceeded {
                current_loss: -self.daily_pnl,
                limit: self.risk_limits.daily_max_loss,
            });
        }

        Ok(())
    }

    /// Get total P&L across all positions
    pub fn total_pnl(&self) -> f64 {
        self.total_realized_pnl + self.total_unrealized_pnl
    }

    /// Get portfolio statistics
    pub fn get_stats(&self) -> PositionStats {
        let mut long_positions = 0;
        let mut short_positions = 0;
        let mut total_market_value = 0.0;

        for position in self.positions.values() {
            if position.is_long() {
                long_positions += 1;
            } else if position.is_short() {
                short_positions += 1;
            }
            total_market_value += position.market_value();
        }

        PositionStats {
            total_positions: self.positions.len(),
            long_positions,
            short_positions,
            total_realized_pnl: self.total_realized_pnl,
            total_unrealized_pnl: self.total_unrealized_pnl,
            total_pnl: self.total_pnl(),
            total_market_value,
            total_commission: self.total_commission,
            high_water_mark: self.high_water_mark,
            max_drawdown: self.max_drawdown,
            current_drawdown: self.current_drawdown,
            daily_pnl: self.daily_pnl,
        }
    }
}

/// Portfolio manager handles multiple strategies
#[derive(Debug)]
pub struct PositionManager {
    /// Position trackers by strategy
    strategy_trackers: HashMap<StrategyId, PositionTracker>,
    /// Global risk limits
    _global_risk_limits: RiskLimits,
    /// Current market prices
    market_prices: HashMap<InstrumentId, Price>,
}

impl PositionManager {
    /// Create a new position manager
    pub fn new(global_risk_limits: RiskLimits) -> Self {
        Self {
            strategy_trackers: HashMap::new(),
            _global_risk_limits: global_risk_limits,
            market_prices: HashMap::new(),
        }
    }

    /// Add a strategy to track
    pub fn add_strategy(&mut self, strategy_id: StrategyId, risk_limits: RiskLimits) {
        let tracker = PositionTracker::new(strategy_id.clone(), risk_limits);
        self.strategy_trackers.insert(strategy_id, tracker);
    }

    /// Apply a fill to a strategy's positions
    pub fn apply_fill(&mut self, fill: &FillEvent) -> Result<(), String> {
        let tracker = self
            .strategy_trackers
            .get_mut(&fill.strategy_id)
            .ok_or_else(|| format!("Strategy {} not found", fill.strategy_id))?;

        // Calculate new quantity after fill
        let current_qty = tracker.get_quantity(fill.instrument_id);
        let fill_qty = match fill.side {
            Side::Bid => fill.quantity.as_i64(),
            Side::Ask => -fill.quantity.as_i64(),
        };
        let new_qty = current_qty + fill_qty;

        // Check risk limits before applying
        tracker
            .check_risk_limits(fill.instrument_id, new_qty)
            .map_err(|violation| format!("Risk violation: {:?}", violation))?;

        // Apply the fill
        tracker.apply_fill(fill);

        Ok(())
    }

    /// Update market prices for all strategies
    pub fn update_market_prices(&mut self, prices: HashMap<InstrumentId, Price>, timestamp: u64) {
        self.market_prices = prices.clone();

        for tracker in self.strategy_trackers.values_mut() {
            tracker.update_market_prices(&prices, timestamp);
        }
    }

    /// Get position tracker for a strategy
    pub fn get_strategy_tracker(&self, strategy_id: &str) -> Option<&PositionTracker> {
        self.strategy_trackers.get(strategy_id)
    }

    /// Get all strategy trackers
    pub fn get_all_trackers(&self) -> &HashMap<StrategyId, PositionTracker> {
        &self.strategy_trackers
    }

    /// Get portfolio-wide statistics
    pub fn get_portfolio_stats(&self) -> PortfolioStats {
        let mut stats = PortfolioStats::default();

        for tracker in self.strategy_trackers.values() {
            let tracker_stats = tracker.get_stats();
            stats.total_strategies += 1;
            stats.total_realized_pnl += tracker_stats.total_realized_pnl;
            stats.total_unrealized_pnl += tracker_stats.total_unrealized_pnl;
            stats.total_commission += tracker_stats.total_commission;
            stats.total_market_value += tracker_stats.total_market_value;

            if tracker_stats.max_drawdown > stats.max_drawdown {
                stats.max_drawdown = tracker_stats.max_drawdown;
            }
        }

        stats.total_pnl = stats.total_realized_pnl + stats.total_unrealized_pnl;
        stats
    }

    /// Reset daily P&L for all strategies
    pub fn reset_daily_pnl(&mut self, timestamp: u64) {
        for tracker in self.strategy_trackers.values_mut() {
            tracker.reset_daily_pnl(timestamp);
        }
    }
}

/// Risk violation types
#[derive(Debug, Clone)]
pub enum RiskViolation {
    /// Position size exceeded for instrument
    PositionSizeExceeded {
        instrument_id: InstrumentId,
        current: u64,
        limit: u64,
    },
    /// Maximum loss exceeded
    MaxLossExceeded { current_loss: f64, limit: f64 },
    /// Daily loss limit exceeded
    DailyLossExceeded { current_loss: f64, limit: f64 },
}

/// Position statistics for a strategy
#[derive(Debug, Clone)]
pub struct PositionStats {
    pub total_positions: usize,
    pub long_positions: usize,
    pub short_positions: usize,
    pub total_realized_pnl: f64,
    pub total_unrealized_pnl: f64,
    pub total_pnl: f64,
    pub total_market_value: f64,
    pub total_commission: f64,
    pub high_water_mark: f64,
    pub max_drawdown: f64,
    pub current_drawdown: f64,
    pub daily_pnl: f64,
}

/// Portfolio-wide statistics
#[derive(Debug, Clone, Default)]
pub struct PortfolioStats {
    pub total_strategies: usize,
    pub total_realized_pnl: f64,
    pub total_unrealized_pnl: f64,
    pub total_pnl: f64,
    pub total_market_value: f64,
    pub total_commission: f64,
    pub max_drawdown: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Quantity;

    #[test]
    fn test_position_creation() {
        let position = Position::new(1);
        assert!(position.is_flat());
        assert!(!position.is_long());
        assert!(!position.is_short());
        assert_eq!(position.abs_quantity(), 0);
        assert_eq!(position.total_pnl(), 0.0);
    }

    #[test]
    fn test_position_long_trade() {
        let mut position = Position::new(1);

        // Buy 10 contracts at $100
        let fill = FillEvent {
            fill_id: 1,
            order_id: 1,
            strategy_id: "test".to_string(),
            instrument_id: 1,
            price: Price::from(100i64),
            quantity: Quantity::from(10u32),
            side: Side::Bid,
            timestamp: 1000,
            commission: 5.0,
            is_maker: false,
        };

        position.apply_fill(&fill);

        assert!(position.is_long());
        assert_eq!(position.quantity, 10);
        assert_eq!(position.avg_price, Price::from(100i64));
        assert_eq!(position.cost_basis, 1005.0); // 10 * 100 + 5 commission
        assert_eq!(position.total_commission, 5.0);
        assert_eq!(position.realized_pnl, 0.0); // No realized P&L yet
    }

    #[test]
    fn test_position_close_with_profit() {
        let mut position = Position::new(1);

        // Buy 10 contracts at $100
        let buy_fill = FillEvent {
            fill_id: 1,
            order_id: 1,
            strategy_id: "test".to_string(),
            instrument_id: 1,
            price: Price::from(100i64),
            quantity: Quantity::from(10u32),
            side: Side::Bid,
            timestamp: 1000,
            commission: 5.0,
            is_maker: false,
        };

        position.apply_fill(&buy_fill);

        // Sell 10 contracts at $110 (profit)
        let sell_fill = FillEvent {
            fill_id: 2,
            order_id: 2,
            strategy_id: "test".to_string(),
            instrument_id: 1,
            price: Price::from(110i64),
            quantity: Quantity::from(10u32),
            side: Side::Ask,
            timestamp: 2000,
            commission: 5.0,
            is_maker: false,
        };

        position.apply_fill(&sell_fill);

        assert!(position.is_flat());
        assert_eq!(position.quantity, 0);
        assert_eq!(position.total_commission, 10.0);
        // Profit: (110 * 10 - 5) - (100 * 10 + 5) = 1095 - 1005 = 90
        assert_eq!(position.realized_pnl, 90.0);
    }

    #[test]
    fn test_position_partial_close() {
        let mut position = Position::new(1);

        // Buy 10 contracts at $100
        let buy_fill = FillEvent {
            fill_id: 1,
            order_id: 1,
            strategy_id: "test".to_string(),
            instrument_id: 1,
            price: Price::from(100i64),
            quantity: Quantity::from(10u32),
            side: Side::Bid,
            timestamp: 1000,
            commission: 5.0,
            is_maker: false,
        };

        position.apply_fill(&buy_fill);

        // Sell 5 contracts at $110 (partial close)
        let sell_fill = FillEvent {
            fill_id: 2,
            order_id: 2,
            strategy_id: "test".to_string(),
            instrument_id: 1,
            price: Price::from(110i64),
            quantity: Quantity::from(5u32),
            side: Side::Ask,
            timestamp: 2000,
            commission: 2.5,
            is_maker: false,
        };

        position.apply_fill(&sell_fill);

        assert!(position.is_long());
        assert_eq!(position.quantity, 5);
        // Realized P&L: (110 * 5 - 2.5) - (100.5 * 5) = 547.5 - 502.5 = 45
        assert!((position.realized_pnl - 45.0).abs() < 0.1);
        assert_eq!(position.total_commission, 7.5);
    }

    #[test]
    fn test_position_tracker() {
        let risk_limits = RiskLimits::default();
        let mut tracker = PositionTracker::new("test_strategy".to_string(), risk_limits);

        // Apply a fill
        let fill = FillEvent {
            fill_id: 1,
            order_id: 1,
            strategy_id: "test_strategy".to_string(),
            instrument_id: 1,
            price: Price::from(100i64),
            quantity: Quantity::from(10u32),
            side: Side::Bid,
            timestamp: 1000,
            commission: 5.0,
            is_maker: false,
        };

        tracker.apply_fill(&fill);

        let stats = tracker.get_stats();
        assert_eq!(stats.total_positions, 1);
        assert_eq!(stats.long_positions, 1);
        assert_eq!(stats.short_positions, 0);
        assert_eq!(stats.total_commission, 5.0);
    }

    #[test]
    fn test_risk_limits() {
        let risk_limits = RiskLimits {
            max_position: 5,
            max_order_size: 10,
            max_loss: 100.0,
            daily_max_loss: 50.0,
            max_orders_per_minute: 10,
        };

        let tracker = PositionTracker::new("test".to_string(), risk_limits);

        // Should fail - exceeds max position
        let result = tracker.check_risk_limits(1, 10);
        assert!(result.is_err());

        // Should pass - within limits
        let result = tracker.check_risk_limits(1, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_position_manager() {
        let global_limits = RiskLimits::default();
        let mut manager = PositionManager::new(global_limits);

        let strategy_limits = RiskLimits::default();
        manager.add_strategy("strategy1".to_string(), strategy_limits);

        let fill = FillEvent {
            fill_id: 1,
            order_id: 1,
            strategy_id: "strategy1".to_string(),
            instrument_id: 1,
            price: Price::from(100i64),
            quantity: Quantity::from(10u32),
            side: Side::Bid,
            timestamp: 1000,
            commission: 5.0,
            is_maker: false,
        };

        let result = manager.apply_fill(&fill);
        assert!(result.is_ok());

        let portfolio_stats = manager.get_portfolio_stats();
        assert_eq!(portfolio_stats.total_strategies, 1);
        assert_eq!(portfolio_stats.total_commission, 5.0);
    }
}
