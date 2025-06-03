//! Market data event types

use crate::core::types::{InstrumentId, Price, Quantity, OrderId};
use crate::core::Side;
use crate::order_book::events::OrderBookEvent;

/// Unified market event type
#[derive(Debug, Clone)]
pub enum MarketEvent {
    /// Order book event (from MBO data)
    OrderBook(OrderBookEvent),
    /// Trade event
    Trade(TradeEvent),
    /// Best bid/offer update
    BBO(BBOUpdate),
    /// Session event
    Session(SessionEvent),
}

impl MarketEvent {
    /// Get the instrument ID for this event
    pub fn instrument_id(&self) -> InstrumentId {
        match self {
            Self::OrderBook(event) => event.instrument_id(),
            Self::Trade(event) => event.instrument_id,
            Self::BBO(event) => event.instrument_id,
            Self::Session(event) => event.instrument_id,
        }
    }
    
    /// Get the timestamp for this event
    pub fn timestamp(&self) -> u64 {
        match self {
            Self::OrderBook(event) => event.timestamp(),
            Self::Trade(event) => event.timestamp,
            Self::BBO(event) => event.timestamp,
            Self::Session(event) => event.timestamp,
        }
    }
    
    /// Check if this is a trade event
    pub fn is_trade(&self) -> bool {
        matches!(self, Self::Trade(_))
    }
    
    /// Check if this is an order book event
    pub fn is_order_book(&self) -> bool {
        matches!(self, Self::OrderBook(_))
    }
}

/// Trade execution event
#[derive(Debug, Clone)]
pub struct TradeEvent {
    pub instrument_id: InstrumentId,
    pub trade_id: u64,
    pub price: Price,
    pub quantity: Quantity,
    pub aggressor_side: Side,
    pub timestamp: u64,
    /// Order IDs involved (if available)
    pub buyer_order_id: Option<OrderId>,
    pub seller_order_id: Option<OrderId>,
}

/// Best bid/offer update
#[derive(Debug, Clone)]
pub struct BBOUpdate {
    pub instrument_id: InstrumentId,
    pub bid_price: Option<Price>,
    pub bid_quantity: Option<Quantity>,
    pub bid_order_count: Option<u32>,
    pub ask_price: Option<Price>,
    pub ask_quantity: Option<Quantity>,
    pub ask_order_count: Option<u32>,
    pub timestamp: u64,
}

impl BBOUpdate {
    /// Get mid price if both bid and ask are available
    pub fn mid_price(&self) -> Option<Price> {
        match (self.bid_price, self.ask_price) {
            (Some(bid), Some(ask)) => {
                Some(Price::from_f64((bid.as_f64() + ask.as_f64()) / 2.0))
            }
            _ => None,
        }
    }
    
    /// Get spread if both bid and ask are available
    pub fn spread(&self) -> Option<f64> {
        match (self.bid_price, self.ask_price) {
            (Some(bid), Some(ask)) => Some(ask.as_f64() - bid.as_f64()),
            _ => None,
        }
    }
}

/// Session-related events
#[derive(Debug, Clone)]
pub struct SessionEvent {
    pub instrument_id: InstrumentId,
    pub event_type: SessionEventType,
    pub timestamp: u64,
}

/// Types of session events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionEventType {
    /// Pre-market session start
    PreMarketOpen,
    /// Regular trading hours start
    MarketOpen,
    /// Regular trading hours end
    MarketClose,
    /// After-hours session end
    AfterMarketClose,
    /// Trading halt
    TradingHalt,
    /// Trading resume
    TradingResume,
    /// End of day
    EndOfDay,
}

/// Market data snapshot (for initialization)
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub instrument_id: InstrumentId,
    pub timestamp: u64,
    pub bids: Vec<(Price, Quantity)>,
    pub asks: Vec<(Price, Quantity)>,
    pub last_trade_price: Option<Price>,
    pub last_trade_quantity: Option<Quantity>,
    pub session_open: Option<Price>,
    pub session_high: Option<Price>,
    pub session_low: Option<Price>,
    pub session_volume: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_event_instrument_id() {
        let trade = TradeEvent {
            instrument_id: 1,
            trade_id: 12345,
            price: Price::from(100i64),
            quantity: Quantity::from(10u32),
            aggressor_side: Side::Bid,
            timestamp: 1000,
            buyer_order_id: None,
            seller_order_id: None,
        };
        
        let event = MarketEvent::Trade(trade);
        assert_eq!(event.instrument_id(), 1);
        assert_eq!(event.timestamp(), 1000);
        assert!(event.is_trade());
        assert!(!event.is_order_book());
    }
    
    #[test]
    fn test_bbo_calculations() {
        let bbo = BBOUpdate {
            instrument_id: 1,
            bid_price: Some(Price::from(100i64)),
            bid_quantity: Some(Quantity::from(10u32)),
            bid_order_count: Some(2),
            ask_price: Some(Price::from(101i64)),
            ask_quantity: Some(Quantity::from(15u32)),
            ask_order_count: Some(3),
            timestamp: 1000,
        };
        
        let mid = bbo.mid_price().unwrap();
        assert_eq!(mid.as_f64(), 100.5);
        
        let spread = bbo.spread().unwrap();
        assert_eq!(spread, 1.0);
    }
}