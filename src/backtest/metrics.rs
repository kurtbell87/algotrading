//! Performance metrics and analytics for backtesting
//!
//! This module provides comprehensive performance metrics calculation,
//! trade analytics, and reporting functionality for backtest results.

use crate::backtest::events::FillEvent;
use crate::backtest::position::{PortfolioStats, PositionStats};
use crate::core::types::{InstrumentId, Price};
use crate::strategy::StrategyId;
use std::collections::{HashMap, VecDeque};

/// Time period for calculations
#[derive(Debug, Clone, Copy)]
pub enum TimePeriod {
    Daily,
    Weekly,
    Monthly,
    Annually,
}

impl TimePeriod {
    /// Convert to annualization factor
    pub fn annualization_factor(&self) -> f64 {
        match self {
            TimePeriod::Daily => 252.0,  // Trading days per year
            TimePeriod::Weekly => 52.0,  // Weeks per year
            TimePeriod::Monthly => 12.0, // Months per year
            TimePeriod::Annually => 1.0, // Already annualized
        }
    }
}

/// Trade record for analytics
#[derive(Debug, Clone)]
pub struct Trade {
    /// Trade identifier
    pub id: u64,
    /// Strategy that made the trade
    pub strategy_id: StrategyId,
    /// Instrument traded
    pub instrument_id: InstrumentId,
    /// Entry price
    pub entry_price: Price,
    /// Exit price
    pub exit_price: Price,
    /// Quantity (positive for long, negative for short)
    pub quantity: i64,
    /// Entry timestamp
    pub entry_time: u64,
    /// Exit timestamp
    pub exit_time: u64,
    /// P&L for this trade
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Commission paid
    pub commission: f64,
    /// Whether trade was profitable
    pub is_winner: bool,
}

impl Trade {
    /// Calculate trade duration in microseconds
    pub fn duration_us(&self) -> u64 {
        self.exit_time - self.entry_time
    }

    /// Calculate trade duration in seconds
    pub fn duration_seconds(&self) -> f64 {
        self.duration_us() as f64 / 1_000_000.0
    }
}

/// Metrics collector for real-time tracking
#[derive(Debug)]
pub struct MetricsCollector {
    /// Completed trades
    trades: Vec<Trade>,
    /// Open positions by instrument
    open_positions: HashMap<InstrumentId, OpenPosition>,
    /// Daily returns
    daily_returns: VecDeque<DailyReturn>,
    /// Equity curve
    equity_curve: Vec<EquityPoint>,
    /// Current equity
    current_equity: f64,
    /// Starting capital
    starting_capital: f64,
    /// High water mark for drawdown
    high_water_mark: f64,
    /// Current drawdown
    current_drawdown: f64,
    /// Maximum drawdown
    max_drawdown: f64,
    /// Last daily reset timestamp
    pub last_daily_reset: u64,
    /// Trade ID counter
    next_trade_id: u64,
}

/// Open position tracking
#[derive(Debug, Clone)]
struct OpenPosition {
    _strategy_id: StrategyId,
    entry_price: Price,
    quantity: i64,
    entry_time: u64,
    entry_fills: Vec<FillEvent>,
}

/// Daily return record
#[derive(Debug, Clone)]
struct DailyReturn {
    _date: u64, // Start of day timestamp
    _starting_equity: f64,
    _ending_equity: f64,
    return_pct: f64,
    _trades_count: usize,
}

/// Equity curve point
#[derive(Debug, Clone)]
pub struct EquityPoint {
    pub timestamp: u64,
    pub equity: f64,
    pub drawdown: f64,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(starting_capital: f64) -> Self {
        Self {
            trades: Vec::new(),
            open_positions: HashMap::new(),
            daily_returns: VecDeque::with_capacity(365),
            equity_curve: vec![EquityPoint {
                timestamp: 0,
                equity: starting_capital,
                drawdown: 0.0,
            }],
            current_equity: starting_capital,
            starting_capital,
            high_water_mark: starting_capital,
            current_drawdown: 0.0,
            max_drawdown: 0.0,
            last_daily_reset: 0,
            next_trade_id: 1,
        }
    }

    /// Process a fill event
    pub fn process_fill(&mut self, fill: &FillEvent) {
        let fill_quantity = match fill.side {
            crate::core::Side::Bid => fill.quantity.as_i64(),
            crate::core::Side::Ask => -fill.quantity.as_i64(),
        };

        // Check if we need to record a trade
        let mut trade_to_record = None;
        let mut should_update_equity = false;

        {
            let position = self
                .open_positions
                .entry(fill.instrument_id)
                .or_insert_with(|| OpenPosition {
                    _strategy_id: fill.strategy_id.clone(),
                    entry_price: fill.price,
                    quantity: 0,
                    entry_time: fill.timestamp,
                    entry_fills: Vec::new(),
                });

            let old_quantity = position.quantity;
            let new_quantity = old_quantity + fill_quantity;

            // Check if this closes or reduces position
            if (old_quantity > 0 && fill_quantity < 0) || (old_quantity < 0 && fill_quantity > 0) {
                let closing_quantity = fill_quantity.abs().min(old_quantity.abs());

                if closing_quantity > 0 {
                    // Calculate P&L for closed portion
                    let avg_entry_price = position.entry_price.as_f64();
                    let exit_price = fill.price.as_f64();

                    let pnl = if old_quantity > 0 {
                        // Closing long
                        (exit_price - avg_entry_price) * closing_quantity as f64
                    } else {
                        // Closing short
                        (avg_entry_price - exit_price) * closing_quantity as f64
                    };

                    let pnl_after_commission = pnl - fill.commission;
                    let return_pct = (pnl / (avg_entry_price * closing_quantity as f64)) * 100.0;

                    // Record completed trade
                    trade_to_record = Some(Trade {
                        id: self.next_trade_id,
                        strategy_id: fill.strategy_id.clone(),
                        instrument_id: fill.instrument_id,
                        entry_price: position.entry_price,
                        exit_price: fill.price,
                        quantity: old_quantity.signum() * closing_quantity,
                        entry_time: position.entry_time,
                        exit_time: fill.timestamp,
                        pnl: pnl_after_commission,
                        return_pct,
                        commission: fill.commission,
                        is_winner: pnl_after_commission > 0.0,
                    });

                    self.current_equity += pnl_after_commission;
                    should_update_equity = true;
                }
            }

            // Update position
            position.quantity = new_quantity;
            position.entry_fills.push(fill.clone());
            let _position_flat = new_quantity == 0;
        }

        // Process trade outside of position borrow
        if let Some(trade) = trade_to_record {
            self.next_trade_id += 1;
            self.trades.push(trade);
        }

        if should_update_equity {
            self.update_equity_curve(fill.timestamp);
        }

        // Remove if flat
        // Check if position is flat and remove it
        if self
            .open_positions
            .get(&fill.instrument_id)
            .map(|p| p.quantity == 0)
            .unwrap_or(false)
        {
            self.open_positions.remove(&fill.instrument_id);
        }
    }

    /// Update equity curve and drawdown
    fn update_equity_curve(&mut self, timestamp: u64) {
        // Update high water mark
        if self.current_equity > self.high_water_mark {
            self.high_water_mark = self.current_equity;
            self.current_drawdown = 0.0;
        } else {
            self.current_drawdown =
                (self.high_water_mark - self.current_equity) / self.high_water_mark;
            if self.current_drawdown > self.max_drawdown {
                self.max_drawdown = self.current_drawdown;
            }
        }

        // Add to equity curve
        self.equity_curve.push(EquityPoint {
            timestamp,
            equity: self.current_equity,
            drawdown: self.current_drawdown,
        });
    }

    /// Reset daily tracking
    pub fn reset_daily(&mut self, timestamp: u64) {
        let starting_equity = self
            .equity_curve
            .last()
            .map(|p| p.equity)
            .unwrap_or(self.starting_capital);

        let ending_equity = self.current_equity;
        let return_pct = ((ending_equity - starting_equity) / starting_equity) * 100.0;

        // Count trades for this day
        let trades_count = self
            .trades
            .iter()
            .filter(|t| t.exit_time >= self.last_daily_reset && t.exit_time < timestamp)
            .count();

        self.daily_returns.push_back(DailyReturn {
            _date: self.last_daily_reset,
            _starting_equity: starting_equity,
            _ending_equity: ending_equity,
            return_pct,
            _trades_count: trades_count,
        });

        // Keep only last year of daily returns
        while self.daily_returns.len() > 365 {
            self.daily_returns.pop_front();
        }

        self.last_daily_reset = timestamp;
    }

    /// Calculate performance metrics
    pub fn calculate_metrics(&self) -> PerformanceMetrics {
        let total_trades = self.trades.len();
        if total_trades == 0 {
            return PerformanceMetrics::default();
        }

        // Basic statistics
        let winning_trades = self.trades.iter().filter(|t| t.is_winner).count();
        let losing_trades = total_trades - winning_trades;
        let win_rate = (winning_trades as f64 / total_trades as f64) * 100.0;

        // P&L statistics
        let gross_profit: f64 = self
            .trades
            .iter()
            .filter(|t| t.is_winner)
            .map(|t| t.pnl)
            .sum();

        let gross_loss: f64 = self
            .trades
            .iter()
            .filter(|t| !t.is_winner)
            .map(|t| t.pnl.abs())
            .sum();

        let total_pnl = self.current_equity - self.starting_capital;
        let total_return = (total_pnl / self.starting_capital) * 100.0;

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            f64::INFINITY
        };

        // Average trade statistics
        let avg_win = if winning_trades > 0 {
            gross_profit / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss = if losing_trades > 0 {
            gross_loss / losing_trades as f64
        } else {
            0.0
        };

        let avg_trade = total_pnl / total_trades as f64;

        // Risk-adjusted returns
        let sharpe = self.calculate_sharpe_ratio();
        let sortino = self.calculate_sortino_ratio();
        let calmar = self.calculate_calmar_ratio();

        // Trade duration
        let avg_trade_duration = if total_trades > 0 {
            let total_duration: u64 = self.trades.iter().map(|t| t.duration_us()).sum();
            (total_duration / total_trades as u64) as f64 / 1_000_000.0
        } else {
            0.0
        };

        PerformanceMetrics {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            profit_factor,
            gross_profit,
            gross_loss,
            total_pnl,
            total_return,
            avg_win,
            avg_loss,
            avg_trade,
            max_drawdown: self.max_drawdown * 100.0,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            calmar_ratio: calmar,
            avg_trade_duration_seconds: avg_trade_duration,
        }
    }

    /// Calculate Sharpe ratio
    fn calculate_sharpe_ratio(&self) -> f64 {
        if self.daily_returns.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self
            .daily_returns
            .iter()
            .map(|r| r.return_pct / 100.0)
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            // Annualized Sharpe ratio (assuming daily returns)
            mean_return * 252.0_f64.sqrt() / std_dev
        } else {
            0.0
        }
    }

    /// Calculate Sortino ratio (downside deviation)
    fn calculate_sortino_ratio(&self) -> f64 {
        if self.daily_returns.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self
            .daily_returns
            .iter()
            .map(|r| r.return_pct / 100.0)
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

        // Calculate downside deviation
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        if negative_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_variance =
            negative_returns.iter().map(|r| r.powi(2)).sum::<f64>() / returns.len() as f64;

        let downside_dev = downside_variance.sqrt();

        if downside_dev > 0.0 {
            // Annualized Sortino ratio
            mean_return * 252.0_f64.sqrt() / downside_dev
        } else {
            0.0
        }
    }

    /// Calculate Calmar ratio (return / max drawdown)
    fn calculate_calmar_ratio(&self) -> f64 {
        if self.max_drawdown > 0.0 {
            let annualized_return = self.calculate_annualized_return();
            annualized_return / (self.max_drawdown * 100.0)
        } else {
            f64::INFINITY
        }
    }

    /// Calculate annualized return
    fn calculate_annualized_return(&self) -> f64 {
        if self.equity_curve.len() < 2 {
            return 0.0;
        }

        let first = &self.equity_curve[0];
        let last = &self.equity_curve[self.equity_curve.len() - 1];

        let total_return = (last.equity - first.equity) / first.equity;
        let duration_years =
            (last.timestamp - first.timestamp) as f64 / (365.25 * 24.0 * 60.0 * 60.0 * 1_000_000.0);

        if duration_years > 0.0 {
            ((1.0 + total_return).powf(1.0 / duration_years) - 1.0) * 100.0
        } else {
            0.0
        }
    }

    /// Get trade history
    pub fn get_trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Get equity curve
    pub fn get_equity_curve(&self) -> &[EquityPoint] {
        &self.equity_curve
    }

    /// Get current statistics
    pub fn get_current_stats(&self) -> CurrentStats {
        CurrentStats {
            current_equity: self.current_equity,
            current_drawdown: self.current_drawdown * 100.0,
            open_positions: self.open_positions.len(),
            total_trades: self.trades.len(),
        }
    }
}

/// Performance metrics summary
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Win rate percentage
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Total gross profit
    pub gross_profit: f64,
    /// Total gross loss
    pub gross_loss: f64,
    /// Total P&L
    pub total_pnl: f64,
    /// Total return percentage
    pub total_return: f64,
    /// Average winning trade
    pub avg_win: f64,
    /// Average losing trade
    pub avg_loss: f64,
    /// Average trade P&L
    pub avg_trade: f64,
    /// Maximum drawdown percentage
    pub max_drawdown: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Average trade duration in seconds
    pub avg_trade_duration_seconds: f64,
}

/// Current statistics
#[derive(Debug, Clone)]
pub struct CurrentStats {
    pub current_equity: f64,
    pub current_drawdown: f64,
    pub open_positions: usize,
    pub total_trades: usize,
}

/// Comprehensive backtest report
#[derive(Debug, Clone)]
pub struct BacktestReport {
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Position statistics
    pub position_stats: PositionStats,
    /// Portfolio statistics
    pub portfolio_stats: PortfolioStats,
    /// Trade history
    pub trades: Vec<Trade>,
    /// Equity curve
    pub equity_curve: Vec<EquityPoint>,
    /// Strategy-specific metrics
    pub strategy_metrics: HashMap<StrategyId, PerformanceMetrics>,
}

impl BacktestReport {
    /// Generate a text summary of the report
    pub fn summary(&self) -> String {
        format!(
            r#"
Backtest Performance Summary
============================

Overview:
---------
Total Trades: {}
Win Rate: {:.2}%
Profit Factor: {:.2}
Total P&L: ${:.2}
Total Return: {:.2}%
Max Drawdown: {:.2}%

Trade Statistics:
-----------------
Winning Trades: {} (${:.2} avg)
Losing Trades: {} (${:.2} avg)
Average Trade: ${:.2}
Average Duration: {:.1} seconds

Risk-Adjusted Returns:
----------------------
Sharpe Ratio: {:.3}
Sortino Ratio: {:.3}
Calmar Ratio: {:.3}

Position Statistics:
--------------------
Total Positions: {}
Long Positions: {}
Short Positions: {}
Total Commission: ${:.2}
"#,
            self.metrics.total_trades,
            self.metrics.win_rate,
            self.metrics.profit_factor,
            self.metrics.total_pnl,
            self.metrics.total_return,
            self.metrics.max_drawdown,
            self.metrics.winning_trades,
            self.metrics.avg_win,
            self.metrics.losing_trades,
            self.metrics.avg_loss,
            self.metrics.avg_trade,
            self.metrics.avg_trade_duration_seconds,
            self.metrics.sharpe_ratio,
            self.metrics.sortino_ratio,
            self.metrics.calmar_ratio,
            self.position_stats.total_positions,
            self.position_stats.long_positions,
            self.position_stats.short_positions,
            self.position_stats.total_commission,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Side;
    use crate::core::types::Quantity;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new(10000.0);
        assert_eq!(collector.current_equity, 10000.0);
        assert_eq!(collector.trades.len(), 0);
        assert_eq!(collector.max_drawdown, 0.0);
    }

    #[test]
    fn test_trade_processing() {
        let mut collector = MetricsCollector::new(10000.0);

        // Open long position
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

        collector.process_fill(&buy_fill);
        assert_eq!(collector.open_positions.len(), 1);

        // Close position with profit
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

        collector.process_fill(&sell_fill);
        assert_eq!(collector.trades.len(), 1);
        assert_eq!(collector.open_positions.len(), 0);

        let trade = &collector.trades[0];
        assert!(trade.is_winner);
        assert_eq!(trade.pnl, 95.0); // (110-100)*10 - 5 = 95 (only exit commission counted)
    }

    #[test]
    fn test_performance_metrics() {
        let mut collector = MetricsCollector::new(10000.0);

        // Add some winning trades
        for i in 0..5 {
            // Buy
            let buy_fill = FillEvent {
                fill_id: i * 2 + 1,
                order_id: i * 2 + 1,
                strategy_id: "test".to_string(),
                instrument_id: 1,
                price: Price::from(100i64),
                quantity: Quantity::from(10u32),
                side: Side::Bid,
                timestamp: i * 1000,
                commission: 1.0,
                is_maker: false,
            };
            collector.process_fill(&buy_fill);

            // Sell with profit
            let sell_fill = FillEvent {
                fill_id: i * 2 + 2,
                order_id: i * 2 + 2,
                strategy_id: "test".to_string(),
                instrument_id: 1,
                price: Price::from(105i64),
                quantity: Quantity::from(10u32),
                side: Side::Ask,
                timestamp: i * 1000 + 500,
                commission: 1.0,
                is_maker: false,
            };
            collector.process_fill(&sell_fill);
        }

        let metrics = collector.calculate_metrics();
        assert_eq!(metrics.total_trades, 5);
        assert_eq!(metrics.winning_trades, 5);
        assert_eq!(metrics.win_rate, 100.0);
        assert!(metrics.profit_factor.is_infinite());
    }

    #[test]
    fn test_drawdown_calculation() {
        let mut collector = MetricsCollector::new(10000.0);

        // Profitable trade
        collector.current_equity = 11000.0;
        collector.update_equity_curve(1000);
        assert_eq!(collector.high_water_mark, 11000.0);
        assert_eq!(collector.current_drawdown, 0.0);

        // Losing trade
        collector.current_equity = 10500.0;
        collector.update_equity_curve(2000);
        assert_eq!(collector.high_water_mark, 11000.0);
        assert!((collector.current_drawdown - 0.0454545).abs() < 0.0001); // 500/11000

        // Another losing trade
        collector.current_equity = 10000.0;
        collector.update_equity_curve(3000);
        assert!((collector.current_drawdown - 0.0909090).abs() < 0.0001); // 1000/11000
        assert_eq!(collector.max_drawdown, collector.current_drawdown);
    }

    #[test]
    fn test_time_period_annualization() {
        assert_eq!(TimePeriod::Daily.annualization_factor(), 252.0);
        assert_eq!(TimePeriod::Weekly.annualization_factor(), 52.0);
        assert_eq!(TimePeriod::Monthly.annualization_factor(), 12.0);
        assert_eq!(TimePeriod::Annually.annualization_factor(), 1.0);
    }
}
