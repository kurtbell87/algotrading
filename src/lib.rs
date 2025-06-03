pub mod core;
pub mod market_data;
pub mod order_book;
pub mod contract_mgmt;
pub mod features;
pub mod strategy;
pub mod backtest;
pub mod strategies;
pub mod python_bridge;

// Re-export commonly used types
pub use core::types::{InstrumentId, Price, Quantity};
pub use core::Side;
pub use order_book::book::Book;
pub use order_book::market::Market;
pub use features::{FeatureExtractor, FeatureConfig, FeatureVector};