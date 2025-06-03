pub mod backtest;
pub mod contract_mgmt;
pub mod core;
pub mod features;
pub mod market_data;
pub mod order_book;
pub mod python_bridge;
pub mod strategies;
pub mod strategy;

// Re-export commonly used types
pub use core::Side;
pub use core::types::{InstrumentId, Price, Quantity};
pub use features::{FeatureConfig, FeatureExtractor, FeatureVector};
pub use order_book::book::Book;
pub use order_book::market::Market;
