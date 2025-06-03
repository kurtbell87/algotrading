use crate::core::types::*;

/// System-wide events
#[derive(Debug, Clone)]
pub enum SystemEvent {
    MarketData(MarketDataEvent),
    OrderBook(OrderBookEvent),
    Execution(ExecutionEvent),
    Risk(RiskEvent),
    Strategy(StrategyEvent),
}

/// Market data events
#[derive(Debug, Clone)]
pub enum MarketDataEvent {
    Connected { source: String },
    Disconnected { source: String },
    Update(MarketUpdate),
}

/// Order book events
#[derive(Debug, Clone)]
pub enum OrderBookEvent {
    Updated {
        instrument_id: InstrumentId,
    },
    Cleared {
        instrument_id: InstrumentId,
    },
    Snapshot {
        instrument_id: InstrumentId,
        depth: BookDepth,
    },
}

/// Execution events
#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    OrderSubmitted { order_id: OrderId },
    OrderCancelled { order_id: OrderId },
    OrderModified { order_id: OrderId },
    OrderFilled(Fill),
    OrderRejected { order_id: OrderId, reason: String },
}

/// Risk events
#[derive(Debug, Clone)]
pub enum RiskEvent {
    LimitBreached {
        limit_type: String,
        current: f64,
        limit: f64,
    },
    PositionUpdate {
        instrument_id: InstrumentId,
        position: Position,
    },
}

/// Strategy events
#[derive(Debug, Clone)]
pub enum StrategyEvent {
    Signal {
        instrument_id: InstrumentId,
        signal: f64,
    },
    StateChange {
        strategy: String,
        state: String,
    },
}
