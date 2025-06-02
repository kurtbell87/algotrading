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