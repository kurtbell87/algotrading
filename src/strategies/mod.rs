//! Example trading strategies
//!
//! This module contains example implementations of common trading strategies
//! to demonstrate the backtesting framework capabilities.

pub mod market_maker;
pub mod mean_reversion;
pub mod trend_following;
pub mod utils;

pub use market_maker::MarketMakerStrategy;
pub use mean_reversion::MeanReversionStrategy;
pub use trend_following::TrendFollowingStrategy;
