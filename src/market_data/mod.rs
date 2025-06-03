pub mod reader;
// pub mod reader_optimized; // Temporarily disabled due to compilation errors
pub mod reader_zero_copy;
pub mod decoder;
pub mod normalizer;
pub mod events;

pub use reader::FileReader;
pub use reader_zero_copy::{ZeroCopyFileReader, FastMarketState, FastFeatureVector, 
                          LAST_PRICE_IDX, SPREAD_IDX, PRICE_RETURN_IDX, VOLUME_IDX, TRADE_COUNT_IDX};
pub use decoder::DbnDecoder;
pub use events::{MarketEvent, TradeEvent, BBOUpdate, SessionEvent};