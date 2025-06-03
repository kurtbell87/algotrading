//! Backtesting engine
//!
//! This module provides a high-performance backtesting framework
//! for evaluating trading strategies on historical data.

pub mod events;
pub mod market_state;
pub mod engine;
pub mod engine_batch;
pub mod engine_fast;
pub mod engine_simple_fast;
pub mod engine_optimized_v2;
pub mod engine_ultra_fast;
// pub mod engine_optimized; // Temporarily disabled due to compilation errors
pub mod execution;
pub mod position;
pub mod metrics;

pub use engine::{BacktestEngine, BacktestConfig, EngineReport, StrategyResult};
pub use events::{BacktestEvent, TimerEvent, OrderUpdateEvent, FillEvent, OrderStatus};
pub use market_state::{MarketState, MarketStateManager};
pub use execution::{ExecutionEngine, LatencyModel, FillModel};
pub use position::{PositionManager, PositionTracker, Position, PositionStats, PortfolioStats, RiskViolation};
pub use metrics::{MetricsCollector, PerformanceMetrics, Trade, BacktestReport, EquityPoint};