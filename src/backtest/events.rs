//! Backtest event types

use crate::core::types::{InstrumentId, Price, Quantity, OrderId};
use crate::core::Side;
use crate::market_data::events::MarketEvent;
use crate::strategy::StrategyId;

/// Events that flow through the backtesting system
#[derive(Debug, Clone)]
pub enum BacktestEvent {
    /// Market data event
    Market(MarketEvent),
    /// Timer event for strategies
    Timer(TimerEvent),
    /// Order status update
    OrderUpdate(OrderUpdateEvent),
    /// Trade fill event
    Fill(FillEvent),
    /// End of data
    EndOfData,
}

impl BacktestEvent {
    /// Get event timestamp
    pub fn timestamp(&self) -> Option<u64> {
        match self {
            Self::Market(event) => Some(event.timestamp()),
            Self::Timer(event) => Some(event.timestamp),
            Self::OrderUpdate(event) => Some(event.timestamp),
            Self::Fill(event) => Some(event.timestamp),
            Self::EndOfData => None,
        }
    }
}

/// Timer event for time-based strategies
#[derive(Debug, Clone)]
pub struct TimerEvent {
    /// Strategy to notify
    pub strategy_id: StrategyId,
    /// Event timestamp
    pub timestamp: u64,
}

/// Order status update event
#[derive(Debug, Clone)]
pub struct OrderUpdateEvent {
    /// Order ID
    pub order_id: OrderId,
    /// Strategy that owns the order
    pub strategy_id: StrategyId,
    /// New order status
    pub status: OrderStatus,
    /// Timestamp
    pub timestamp: u64,
    /// Optional message
    pub message: Option<String>,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    /// Order accepted by exchange
    Accepted,
    /// Order partially filled
    PartiallyFilled,
    /// Order completely filled
    Filled,
    /// Order cancelled
    Cancelled,
    /// Order rejected
    Rejected,
    /// Order expired
    Expired,
}

/// Trade fill event
#[derive(Debug, Clone)]
pub struct FillEvent {
    /// Unique fill ID
    pub fill_id: u64,
    /// Order that was filled
    pub order_id: OrderId,
    /// Strategy that owns the order
    pub strategy_id: StrategyId,
    /// Instrument
    pub instrument_id: InstrumentId,
    /// Fill price
    pub price: Price,
    /// Fill quantity
    pub quantity: Quantity,
    /// Side of the fill
    pub side: Side,
    /// Timestamp
    pub timestamp: u64,
    /// Commission paid
    pub commission: f64,
    /// Whether this was a maker or taker fill
    pub is_maker: bool,
}

impl FillEvent {
    /// Calculate total cost including commission
    pub fn total_cost(&self) -> f64 {
        let notional = self.price.as_f64() * self.quantity.as_f64();
        match self.side {
            Side::Bid => notional + self.commission, // Buying costs more
            Side::Ask => notional - self.commission, // Selling receives less
        }
    }
}

/// Event priority for ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventPriority {
    /// Market data has highest priority
    MarketData = 0,
    /// Order updates
    OrderUpdate = 1,
    /// Fills
    Fill = 2,
    /// Timer events have lowest priority
    Timer = 3,
}

impl BacktestEvent {
    /// Get event priority
    pub fn priority(&self) -> EventPriority {
        match self {
            Self::Market(_) => EventPriority::MarketData,
            Self::OrderUpdate(_) => EventPriority::OrderUpdate,
            Self::Fill(_) => EventPriority::Fill,
            Self::Timer(_) => EventPriority::Timer,
            Self::EndOfData => EventPriority::Timer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fill_total_cost() {
        let fill = FillEvent {
            fill_id: 1,
            order_id: 123,
            strategy_id: "test".to_string(),
            instrument_id: 1,
            price: Price::from(100i64),
            quantity: Quantity::from(10u32),
            side: Side::Bid,
            timestamp: 1000,
            commission: 1.0,
            is_maker: true,
        };
        
        // Buy 10 @ 100 = 1000 + 1 commission = 1001
        assert_eq!(fill.total_cost(), 1001.0);
        
        // Sell side
        let sell_fill = FillEvent {
            side: Side::Ask,
            ..fill
        };
        
        // Sell 10 @ 100 = 1000 - 1 commission = 999
        assert_eq!(sell_fill.total_cost(), 999.0);
    }
    
    #[test]
    fn test_event_priority() {
        assert!(EventPriority::MarketData < EventPriority::OrderUpdate);
        assert!(EventPriority::OrderUpdate < EventPriority::Fill);
        assert!(EventPriority::Fill < EventPriority::Timer);
    }
}