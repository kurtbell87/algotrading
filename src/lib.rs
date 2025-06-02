pub mod core;
pub mod market_data;
pub mod order_book;
pub mod contract_mgmt;
pub mod features;

// Re-export commonly used types
pub use core::{InstrumentId, Price, Quantity, Side};
pub use order_book::{Book, Market};
pub use features::{FeatureExtractor, FeatureConfig, FeatureVector};