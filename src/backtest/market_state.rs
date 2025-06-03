//! Market state management for backtesting

use crate::core::types::{InstrumentId, Price, Quantity};
use crate::order_book::book::Book;
use crate::order_book::events::OrderBookEvent;
use crate::market_data::events::{MarketEvent, TradeEvent, BBOUpdate, SessionEvent};
use crate::strategy::context::{RecentTrade, SessionStatistics};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

/// Manages market state during backtesting
pub struct MarketStateManager {
    /// Order books by instrument
    order_books: HashMap<InstrumentId, Arc<RwLock<Book>>>,
    /// Recent trades by instrument
    recent_trades: HashMap<InstrumentId, VecDeque<RecentTrade>>,
    /// Session statistics by instrument
    session_stats: HashMap<InstrumentId, SessionStatistics>,
    /// Maximum trades to keep per instrument
    max_trades_per_instrument: usize,
}

impl MarketStateManager {
    /// Create a new market state manager
    pub fn new() -> Self {
        Self {
            order_books: HashMap::new(),
            recent_trades: HashMap::new(),
            session_stats: HashMap::new(),
            max_trades_per_instrument: 1000,
        }
    }
    
    /// Process a market event and update state
    pub fn process_event(&mut self, event: &MarketEvent) {
        match event {
            MarketEvent::OrderBook(book_event) => {
                self.process_order_book_event(book_event);
            }
            MarketEvent::Trade(trade_event) => {
                self.process_trade_event(trade_event);
            }
            MarketEvent::BBO(bbo_update) => {
                self.process_bbo_update(bbo_update);
            }
            MarketEvent::Session(session_event) => {
                self.process_session_event(session_event);
            }
        }
    }
    
    /// Process order book event
    fn process_order_book_event(&mut self, event: &OrderBookEvent) {
        let instrument_id = event.instrument_id();
        
        // Get or create order book
        let _book = self.order_books
            .entry(instrument_id)
            .or_insert_with(|| Arc::new(RwLock::new(Book::new())));
        
        // Convert OrderBookEvent to MBO message format and apply
        // For now, we'll skip this as it requires conversion logic
        // TODO: Implement OrderBookEvent to MBO conversion
    }
    
    /// Process trade event
    fn process_trade_event(&mut self, event: &TradeEvent) {
        // Add to recent trades
        let trades = self.recent_trades
            .entry(event.instrument_id)
            .or_insert_with(|| VecDeque::with_capacity(self.max_trades_per_instrument));
        
        trades.push_back(RecentTrade {
            price: event.price,
            quantity: event.quantity,
            aggressor_side: event.aggressor_side,
            timestamp: event.timestamp,
        });
        
        // Limit size
        while trades.len() > self.max_trades_per_instrument {
            trades.pop_front();
        }
        
        // Update session statistics
        self.update_session_stats_with_trade(event);
    }
    
    /// Process BBO update
    fn process_bbo_update(&mut self, _update: &BBOUpdate) {
        // BBO updates are handled through order book events in our system
        // This is here for compatibility with other data sources
    }
    
    /// Process session event
    fn process_session_event(&mut self, event: &SessionEvent) {
        use crate::market_data::events::SessionEventType;
        
        match event.event_type {
            SessionEventType::MarketOpen => {
                // Reset session stats for new day
                self.session_stats.insert(event.instrument_id, SessionStatistics::default());
            }
            SessionEventType::EndOfDay => {
                // Clear trades for next day
                self.recent_trades.remove(&event.instrument_id);
            }
            _ => {}
        }
    }
    
    /// Update session statistics with trade
    fn update_session_stats_with_trade(&mut self, trade: &TradeEvent) {
        let stats = self.session_stats
            .entry(trade.instrument_id)
            .or_insert_with(SessionStatistics::default);
        // Update open
        if stats.open.is_none() {
            stats.open = Some(trade.price);
        }
        
        // Update high
        match stats.high {
            Some(high) if trade.price.as_f64() > high.as_f64() => {
                stats.high = Some(trade.price);
            }
            None => stats.high = Some(trade.price),
            _ => {}
        }
        
        // Update low
        match stats.low {
            Some(low) if trade.price.as_f64() < low.as_f64() => {
                stats.low = Some(trade.price);
            }
            None => stats.low = Some(trade.price),
            _ => {}
        }
        
        // Update close
        stats.close = Some(trade.price);
        
        // Update volume
        stats.volume += trade.quantity.as_f64() as u64;
        stats.trade_count += 1;
        
        // Update VWAP
        // VWAP = Σ(Price × Volume) / Σ(Volume)
        // We'll store cumulative values and calculate on demand
    }
    
    /// Get order book for an instrument
    pub fn get_order_book(&self, instrument_id: InstrumentId) -> Option<Arc<RwLock<Book>>> {
        self.order_books.get(&instrument_id).cloned()
    }
    
    /// Get recent trades for an instrument
    pub fn get_recent_trades(&self, instrument_id: InstrumentId, count: usize) -> Vec<RecentTrade> {
        self.recent_trades
            .get(&instrument_id)
            .map(|trades| {
                trades.iter()
                    .rev()
                    .take(count)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Get session statistics
    pub fn get_session_stats(&self, instrument_id: InstrumentId) -> Option<SessionStatistics> {
        self.session_stats.get(&instrument_id).cloned()
    }
    
    /// Create a market state snapshot
    pub fn create_snapshot(&self, instrument_id: InstrumentId) -> Option<MarketState> {
        let book = self.get_order_book(instrument_id)?;
        let book = book.read().unwrap();
        
        let (best_bid, best_ask) = book.bbo();
        
        let bid_data = best_bid.as_ref().map(|l| (Price::new(l.price), Quantity::new(l.size)));
        let ask_data = best_ask.as_ref().map(|l| (Price::new(l.price), Quantity::new(l.size)));
        
        let mid_price = match (&best_bid, &best_ask) {
            (Some(bid), Some(ask)) => {
                let bid_price = Price::new(bid.price);
                let ask_price = Price::new(ask.price);
                Some(Price::from_f64((bid_price.as_f64() + ask_price.as_f64()) / 2.0))
            }
            _ => None,
        };
        
        let spread = match (&best_bid, &best_ask) {
            (Some(bid), Some(ask)) => {
                let bid_price = Price::new(bid.price);
                let ask_price = Price::new(ask.price);
                Some(ask_price.as_f64() - bid_price.as_f64())
            }
            _ => None,
        };
        
        Some(MarketState {
            instrument_id,
            best_bid: bid_data,
            best_ask: ask_data,
            mid_price,
            spread,
            session_stats: self.session_stats.get(&instrument_id).cloned(),
        })
    }
}

impl Default for MarketStateManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of market state at a point in time
#[derive(Debug, Clone)]
pub struct MarketState {
    pub instrument_id: InstrumentId,
    pub best_bid: Option<(Price, Quantity)>,
    pub best_ask: Option<(Price, Quantity)>,
    pub mid_price: Option<Price>,
    pub spread: Option<f64>,
    pub session_stats: Option<SessionStatistics>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Side;
    
    #[test]
    fn test_trade_processing() {
        let mut manager = MarketStateManager::new();
        
        let trade = TradeEvent {
            instrument_id: 1,
            trade_id: 123,
            price: Price::from(100i64),
            quantity: Quantity::from(10u32),
            aggressor_side: Side::Bid,
            timestamp: 1000,
            buyer_order_id: None,
            seller_order_id: None,
        };
        
        manager.process_event(&MarketEvent::Trade(trade));
        
        let trades = manager.get_recent_trades(1, 10);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, Price::from(100i64));
        
        let stats = manager.get_session_stats(1).unwrap();
        assert_eq!(stats.open, Some(Price::from(100i64)));
        assert_eq!(stats.high, Some(Price::from(100i64)));
        assert_eq!(stats.low, Some(Price::from(100i64)));
        assert_eq!(stats.close, Some(Price::from(100i64)));
        assert_eq!(stats.volume, 10);
        assert_eq!(stats.trade_count, 1);
    }
    
    #[test]
    fn test_order_book_creation() {
        let mut manager = MarketStateManager::new();
        
        let event = OrderBookEvent::OrderAdded {
            instrument_id: 1,
            publisher_id: 1,
            order_id: 123,
            side: Side::Bid,
            price: Price::from(100i64),
            quantity: Quantity::from(10u32),
            timestamp: 1000,
        };
        
        manager.process_event(&MarketEvent::OrderBook(event));
        
        let book = manager.get_order_book(1);
        assert!(book.is_some());
    }
}