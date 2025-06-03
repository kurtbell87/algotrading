//! Backtesting engine
//!
//! This module provides a high-performance backtesting framework
//! for evaluating trading strategies on historical data.

pub mod engine;
pub mod engine_batch;
pub mod engine_fast;
pub mod engine_optimized_v2;
pub mod engine_simple_fast;
pub mod engine_ultra_fast;
pub mod events;
pub mod market_state;
// pub mod engine_optimized; // Temporarily disabled due to compilation errors
pub mod execution;
pub mod metrics;
pub mod position;

pub use engine::{BacktestConfig, BacktestEngine, EngineReport, StrategyResult};
pub use events::{BacktestEvent, FillEvent, OrderStatus, OrderUpdateEvent, TimerEvent};
pub use execution::{ExecutionEngine, FillModel, LatencyModel};
pub use market_state::{MarketState, MarketStateManager};
pub use metrics::{BacktestReport, EquityPoint, MetricsCollector, PerformanceMetrics, Trade};
pub use position::{
    PortfolioStats, Position, PositionManager, PositionStats, PositionTracker, RiskViolation,
};
