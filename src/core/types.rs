use std::fmt;

/// Instrument identifier
pub type InstrumentId = u32;

/// Publisher identifier  
pub type PublisherId = u8;

/// Order identifier
pub type OrderId = u64;

/// Price in fixed point representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Price(pub i64);

impl Price {
    pub fn new(value: i64) -> Self {
        Self(value)
    }
    
    pub fn as_f64(&self) -> f64 {
        self.0 as f64 / 1e9
    }
}

impl From<i64> for Price {
    fn from(value: i64) -> Self {
        Self(value * 1_000_000_000) // Convert to fixed point
    }
}

impl From<u64> for Price {
    fn from(value: u64) -> Self {
        Self((value * 1_000_000_000) as i64)
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}", self.as_f64())
    }
}

/// Quantity/Size
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Quantity(pub u32);

impl Quantity {
    pub fn new(value: u32) -> Self {
        Self(value)
    }
    
    pub fn as_f64(&self) -> f64 {
        self.0 as f64
    }
}

impl From<u32> for Quantity {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<u64> for Quantity {
    fn from(value: u64) -> Self {
        Self(value as u32)
    }
}

/// Side of the market
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

/// A single price level in the order book
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriceLevel {
    pub price: Price,
    pub quantity: Quantity,
    pub order_count: u32,
}

/// Book depth snapshot
#[derive(Debug, Clone)]
pub struct BookDepth {
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub timestamp: u64,
}

/// Trade information
#[derive(Debug, Clone)]
pub struct Trade {
    pub instrument_id: InstrumentId,
    pub price: Price,
    pub quantity: Quantity,
    pub side: Side,
    pub timestamp: u64,
}

/// Order to be submitted
#[derive(Debug, Clone)]
pub struct Order {
    pub instrument_id: InstrumentId,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub order_type: OrderType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
}

/// Fill/execution information
#[derive(Debug, Clone)]
pub struct Fill {
    pub order_id: OrderId,
    pub instrument_id: InstrumentId,
    pub price: Price,
    pub quantity: Quantity,
    pub side: Side,
    pub timestamp: u64,
}

/// Position information
#[derive(Debug, Clone, Default)]
pub struct Position {
    pub instrument_id: InstrumentId,
    pub quantity: i32, // Positive for long, negative for short
    pub avg_price: Price,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
}

/// Risk limits
#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub max_position_size: u32,
    pub max_order_size: u32,
    pub max_daily_loss: f64,
    pub max_drawdown: f64,
}

/// Market update event
#[derive(Debug, Clone)]
pub enum MarketUpdate {
    OrderBook(BookUpdate),
    Trade(Trade),
}

/// Order book update
#[derive(Debug, Clone)]
pub struct BookUpdate {
    pub instrument_id: InstrumentId,
    pub update_type: BookUpdateType,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub enum BookUpdateType {
    Add { order_id: OrderId, side: Side, price: Price, quantity: Quantity },
    Modify { order_id: OrderId, new_quantity: Quantity },
    Cancel { order_id: OrderId },
    Clear,
}