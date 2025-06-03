use crate::core::{InstrumentId, PublisherId, OrderId, Price, Quantity, Side};

/// Order book specific events
#[derive(Debug, Clone)]
pub enum OrderBookEvent {
    /// Order added to book
    OrderAdded {
        instrument_id: InstrumentId,
        publisher_id: PublisherId,
        order_id: OrderId,
        side: Side,
        price: Price,
        quantity: Quantity,
        timestamp: u64,
    },
    
    /// Order modified in book
    OrderModified {
        instrument_id: InstrumentId,
        publisher_id: PublisherId,
        order_id: OrderId,
        new_quantity: Quantity,
        timestamp: u64,
    },
    
    /// Order cancelled from book
    OrderCancelled {
        instrument_id: InstrumentId,
        publisher_id: PublisherId,
        order_id: OrderId,
        timestamp: u64,
    },
    
    /// Book cleared
    BookCleared {
        instrument_id: InstrumentId,
        publisher_id: PublisherId,
        timestamp: u64,
    },
    
    /// Best bid/ask changed
    BBOChanged {
        instrument_id: InstrumentId,
        publisher_id: PublisherId,
        bid_price: Option<Price>,
        bid_quantity: Option<Quantity>,
        ask_price: Option<Price>,
        ask_quantity: Option<Quantity>,
        timestamp: u64,
    },
}

impl OrderBookEvent {
    /// Get instrument ID
    pub fn instrument_id(&self) -> InstrumentId {
        match self {
            Self::OrderAdded { instrument_id, .. } => *instrument_id,
            Self::OrderModified { instrument_id, .. } => *instrument_id,
            Self::OrderCancelled { instrument_id, .. } => *instrument_id,
            Self::BookCleared { instrument_id, .. } => *instrument_id,
            Self::BBOChanged { instrument_id, .. } => *instrument_id,
        }
    }
    
    /// Get timestamp
    pub fn timestamp(&self) -> u64 {
        match self {
            Self::OrderAdded { timestamp, .. } => *timestamp,
            Self::OrderModified { timestamp, .. } => *timestamp,
            Self::OrderCancelled { timestamp, .. } => *timestamp,
            Self::BookCleared { timestamp, .. } => *timestamp,
            Self::BBOChanged { timestamp, .. } => *timestamp,
        }
    }
    
    /// Get publisher ID
    pub fn publisher_id(&self) -> PublisherId {
        match self {
            Self::OrderAdded { publisher_id, .. } => *publisher_id,
            Self::OrderModified { publisher_id, .. } => *publisher_id,
            Self::OrderCancelled { publisher_id, .. } => *publisher_id,
            Self::BookCleared { publisher_id, .. } => *publisher_id,
            Self::BBOChanged { publisher_id, .. } => *publisher_id,
        }
    }
}