pub mod reader;
// pub mod reader_optimized; // Temporarily disabled due to compilation errors
pub mod decoder;
pub mod events;
pub mod normalizer;
pub mod reader_zero_copy;

pub use decoder::DbnDecoder;
pub use events::{BBOUpdate, MarketEvent, SessionEvent, TradeEvent};
pub use reader::FileReader;
pub use reader_zero_copy::{
    FastFeatureVector, FastMarketState, LAST_PRICE_IDX, PRICE_RETURN_IDX, SPREAD_IDX,
    TRADE_COUNT_IDX, VOLUME_IDX, ZeroCopyFileReader,
};
