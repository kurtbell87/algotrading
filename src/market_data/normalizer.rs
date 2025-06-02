use crate::core::{
    BookUpdate, BookUpdateType, MarketUpdate,
    Price, Quantity, Side, Trade,
};
use databento::dbn::{
    enums::{Action, Side as DbnSide},
    record::MboMsg,
};

/// Convert MBO message to internal MarketUpdate format
pub fn mbo_to_market_update(msg: &MboMsg) -> MarketUpdate {
    match msg.action() {
        Ok(Action::Trade) => MarketUpdate::Trade(mbo_to_trade(msg)),
        _ => MarketUpdate::OrderBook(mbo_to_book_update(msg)),
    }
}

/// Convert MBO message to Trade
fn mbo_to_trade(msg: &MboMsg) -> Trade {
    Trade {
        instrument_id: msg.hd.instrument_id,
        price: Price::new(msg.price),
        quantity: Quantity::new(msg.size),
        side: dbn_side_to_core(msg.side().unwrap_or(DbnSide::None)),
        timestamp: msg.hd.ts_event,
    }
}

/// Convert MBO message to BookUpdate
fn mbo_to_book_update(msg: &MboMsg) -> BookUpdate {
    let update_type = match msg.action().unwrap_or(Action::Add) {
        Action::Add => BookUpdateType::Add {
            order_id: msg.order_id,
            side: dbn_side_to_core(msg.side().unwrap_or(DbnSide::None)),
            price: Price::new(msg.price),
            quantity: Quantity::new(msg.size),
        },
        Action::Modify => BookUpdateType::Modify {
            order_id: msg.order_id,
            new_quantity: Quantity::new(msg.size),
        },
        Action::Cancel => BookUpdateType::Cancel {
            order_id: msg.order_id,
        },
        Action::Clear => BookUpdateType::Clear,
        _ => BookUpdateType::Clear, // Default to clear for unknown actions
    };

    BookUpdate {
        instrument_id: msg.hd.instrument_id,
        update_type,
        timestamp: msg.hd.ts_event,
    }
}

/// Convert Databento Side to core Side
fn dbn_side_to_core(side: DbnSide) -> Side {
    match side {
        DbnSide::Bid => Side::Bid,
        DbnSide::Ask => Side::Ask,
        DbnSide::None => Side::Bid, // Default to Bid for None
    }
}