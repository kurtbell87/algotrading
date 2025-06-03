//! Trading strategy framework
//!
//! This module defines the core traits and types for implementing trading strategies
//! in a backtesting environment.

pub mod traits;
pub mod context;
pub mod output;

pub use traits::{Strategy, StrategyConfig};
pub use context::{StrategyContext, MarketStateView};
pub use output::{StrategyOutput, OrderRequest, OrderUpdate, StrategyMetrics};

use std::fmt;

/// Unique strategy identifier
pub type StrategyId = String;

/// Strategy initialization error
#[derive(Debug)]
pub enum StrategyError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Missing required data
    MissingData(String),
    /// Risk limit exceeded
    RiskLimitExceeded(String),
    /// Feature extraction error
    FeatureError(String),
    /// Strategy initialization error
    InitializationError(String),
    /// Invalid order error
    InvalidOrder(String),
    /// Other errors
    Other(String),
}

impl fmt::Display for StrategyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::MissingData(msg) => write!(f, "Missing data: {}", msg),
            Self::RiskLimitExceeded(msg) => write!(f, "Risk limit exceeded: {}", msg),
            Self::FeatureError(msg) => write!(f, "Feature error: {}", msg),
            Self::InitializationError(msg) => write!(f, "Initialization error: {}", msg),
            Self::InvalidOrder(msg) => write!(f, "Invalid order: {}", msg),
            Self::Other(msg) => write!(f, "Strategy error: {}", msg),
        }
    }
}

impl std::error::Error for StrategyError {}

impl From<StrategyError> for String {
    fn from(err: StrategyError) -> String {
        err.to_string()
    }
}

impl From<pyo3::PyErr> for StrategyError {
    fn from(err: pyo3::PyErr) -> Self {
        StrategyError::Other(format!("Python error: {}", err))
    }
}

/// Strategy state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyState {
    /// Strategy is initializing
    Initializing,
    /// Strategy is active and can trade
    Active,
    /// Strategy is paused (no new orders)
    Paused,
    /// Strategy is closing positions
    Closing,
    /// Strategy is stopped
    Stopped,
    /// Strategy encountered an error
    Error,
}

/// Signal type for strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// No signal
    None,
    /// Buy signal with strength (0-1)
    Long(f64),
    /// Sell signal with strength (0-1)
    Short(f64),
    /// Exit all positions
    Exit,
    /// Close long positions only
    ExitLong,
    /// Close short positions only
    ExitShort,
}

/// Time in force for orders
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good till cancelled
    GTC,
    /// Immediate or cancel
    IOC,
    /// Fill or kill
    FOK,
    /// Day order
    Day,
}

/// Order side with more detail than core Side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    /// Buy to open or add to long
    Buy,
    /// Sell to close long
    Sell,
    /// Sell to open short
    SellShort,
    /// Buy to close short
    BuyCover,
}