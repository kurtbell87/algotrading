//! Strategy execution context

use crate::core::types::{InstrumentId, OrderId, Price, Quantity};
use crate::features::{FeaturePosition, RiskLimits};
use crate::order_book::book::{Book, LevelSummary};
use crate::strategy::StrategyId;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

/// Information about a pending order
#[derive(Debug, Clone)]
pub struct PendingOrder {
    pub order_id: OrderId,
    pub instrument_id: InstrumentId,
    pub side: crate::core::Side,
    pub order_type: crate::core::types::OrderType,
    pub price: Option<Price>,
    pub quantity: Quantity,
    pub timestamp: u64,
}

/// Recent trade information
#[derive(Debug, Clone)]
pub struct RecentTrade {
    pub price: Price,
    pub quantity: Quantity,
    pub aggressor_side: crate::core::Side,
    pub timestamp: u64,
}

/// Session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStatistics {
    pub open: Option<Price>,
    pub high: Option<Price>,
    pub low: Option<Price>,
    pub close: Option<Price>,
    pub volume: u64,
    pub trade_count: u32,
    pub vwap: Option<Price>,
}

/// Strategy execution context provided to strategies
#[derive(Debug, Clone)]
pub struct StrategyContext {
    /// Strategy identifier
    pub strategy_id: StrategyId,
    /// Current timestamp
    pub current_time: u64,
    /// Current position
    pub position: FeaturePosition,
    /// Pending orders
    pub pending_orders: Vec<PendingOrder>,
    /// Market state view
    pub market_state: MarketStateView,
    /// Risk limits
    pub risk_limits: RiskLimits,
    /// Whether we're in backtesting mode
    pub is_backtesting: bool,
}

impl StrategyContext {
    /// Create a new strategy context
    pub fn new(
        strategy_id: StrategyId,
        current_time: u64,
        position: FeaturePosition,
        risk_limits: RiskLimits,
        is_backtesting: bool,
    ) -> Self {
        Self {
            strategy_id,
            current_time,
            position,
            pending_orders: Vec::new(),
            market_state: MarketStateView::new(),
            risk_limits,
            is_backtesting,
        }
    }

    /// Check if we have any pending orders
    pub fn has_pending_orders(&self) -> bool {
        !self.pending_orders.is_empty()
    }

    /// Get pending orders for an instrument
    pub fn pending_orders_for(&self, instrument_id: InstrumentId) -> Vec<&PendingOrder> {
        self.pending_orders
            .iter()
            .filter(|o| o.instrument_id == instrument_id)
            .collect()
    }

    /// Calculate available buying power
    pub fn available_buying_power(&self) -> i64 {
        let used = self.position.quantity.abs();
        self.risk_limits.max_position - used
    }

    /// Check if we can place an order of given size
    pub fn can_place_order(&self, size: i64) -> bool {
        let new_position = self.position.quantity + size;
        new_position.abs() <= self.risk_limits.max_position
    }
}

/// View of current market state
#[derive(Debug, Clone)]
pub struct MarketStateView {
    /// Order books by instrument
    order_books: Arc<RwLock<Vec<(InstrumentId, Arc<RwLock<Book>>)>>>,
    /// Recent trades by instrument  
    recent_trades: Arc<RwLock<Vec<(InstrumentId, VecDeque<RecentTrade>)>>>,
    /// Session statistics by instrument
    session_stats: Arc<RwLock<Vec<(InstrumentId, SessionStatistics)>>>,
}

impl MarketStateView {
    /// Create a new market state view
    pub fn new() -> Self {
        Self {
            order_books: Arc::new(RwLock::new(Vec::new())),
            recent_trades: Arc::new(RwLock::new(Vec::new())),
            session_stats: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add an order book reference
    pub fn add_order_book(&mut self, instrument_id: InstrumentId, book: Arc<RwLock<Book>>) {
        let mut books = self.order_books.write().unwrap();
        // Remove existing if any
        books.retain(|(id, _)| *id != instrument_id);
        books.push((instrument_id, book));
    }

    /// Get best bid for an instrument
    pub fn best_bid(&self, instrument_id: InstrumentId) -> Option<LevelSummary> {
        let books = self.order_books.read().unwrap();
        for (id, book) in books.iter() {
            if *id == instrument_id {
                let book = book.read().unwrap();
                return book.bbo().0;
            }
        }
        None
    }

    /// Get best ask for an instrument
    pub fn best_ask(&self, instrument_id: InstrumentId) -> Option<LevelSummary> {
        let books = self.order_books.read().unwrap();
        for (id, book) in books.iter() {
            if *id == instrument_id {
                let book = book.read().unwrap();
                return book.bbo().1;
            }
        }
        None
    }

    /// Get mid price
    pub fn mid_price(&self, instrument_id: InstrumentId) -> Option<Price> {
        let bid = self.best_bid(instrument_id)?;
        let ask = self.best_ask(instrument_id)?;
        Some(Price::from_f64(
            (Price::new(bid.price).as_f64() + Price::new(ask.price).as_f64()) / 2.0,
        ))
    }

    /// Get spread
    pub fn spread(&self, instrument_id: InstrumentId) -> Option<f64> {
        let bid = self.best_bid(instrument_id)?;
        let ask = self.best_ask(instrument_id)?;
        Some(Price::new(ask.price).as_f64() - Price::new(bid.price).as_f64())
    }

    /// Add recent trade
    pub fn add_trade(&mut self, instrument_id: InstrumentId, trade: RecentTrade) {
        let mut trades = self.recent_trades.write().unwrap();

        // Find or create trade list for instrument
        let trade_list = trades
            .iter_mut()
            .find(|(id, _)| *id == instrument_id)
            .map(|(_, list)| list);

        if let Some(list) = trade_list {
            list.push_back(trade);
            // Keep only last 1000 trades
            while list.len() > 1000 {
                list.pop_front();
            }
        } else {
            let mut list = VecDeque::with_capacity(1000);
            list.push_back(trade);
            trades.push((instrument_id, list));
        }
    }

    /// Get recent trades
    pub fn recent_trades(&self, instrument_id: InstrumentId, count: usize) -> Vec<RecentTrade> {
        let trades = self.recent_trades.read().unwrap();
        for (id, list) in trades.iter() {
            if *id == instrument_id {
                return list.iter().rev().take(count).cloned().collect();
            }
        }
        Vec::new()
    }

    /// Update session statistics
    pub fn update_session_stats(&mut self, instrument_id: InstrumentId, stats: SessionStatistics) {
        let mut all_stats = self.session_stats.write().unwrap();

        // Find or update
        if let Some((_, existing)) = all_stats.iter_mut().find(|(id, _)| *id == instrument_id) {
            *existing = stats;
        } else {
            all_stats.push((instrument_id, stats));
        }
    }

    /// Get session statistics
    pub fn session_stats(&self, instrument_id: InstrumentId) -> Option<SessionStatistics> {
        let stats = self.session_stats.read().unwrap();
        stats
            .iter()
            .find(|(id, _)| *id == instrument_id)
            .map(|(_, stats)| stats.clone())
    }
}

impl Default for MarketStateView {
    fn default() -> Self {
        Self::new()
    }
}
