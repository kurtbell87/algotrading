//! Strategy output types

use crate::core::types::{InstrumentId, Price, Quantity, OrderId};
use crate::core::Side;
use crate::strategy::{OrderSide, TimeInForce, StrategyId};
use std::collections::HashMap;

/// Order request from strategy
#[derive(Debug, Clone)]
pub struct OrderRequest {
    /// Strategy that generated this order
    pub strategy_id: StrategyId,
    /// Instrument to trade
    pub instrument_id: InstrumentId,
    /// Order side
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Quantity to trade
    pub quantity: Quantity,
    /// Price (for limit orders)
    pub price: Option<Price>,
    /// Time in force
    pub time_in_force: TimeInForce,
    /// Client order ID (optional)
    pub client_order_id: Option<String>,
    /// Order tags for analysis
    pub tags: HashMap<String, String>,
}

/// Order type with more options than core
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    /// Market order
    Market,
    /// Limit order
    Limit,
    /// Stop order (becomes market when triggered)
    Stop,
    /// Stop limit order
    StopLimit,
    /// Pegged order (relative to BBO)
    Pegged { offset: i64 },
}

impl OrderRequest {
    /// Create a market order
    pub fn market_order(
        strategy_id: StrategyId,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
    ) -> Self {
        Self {
            strategy_id,
            instrument_id,
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            time_in_force: TimeInForce::IOC,
            client_order_id: None,
            tags: HashMap::new(),
        }
    }
    
    /// Create a limit order
    pub fn limit_order(
        strategy_id: StrategyId,
        instrument_id: InstrumentId,
        side: OrderSide,
        price: Price,
        quantity: Quantity,
    ) -> Self {
        Self {
            strategy_id,
            instrument_id,
            side,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            time_in_force: TimeInForce::GTC,
            client_order_id: None,
            tags: HashMap::new(),
        }
    }
    
    /// Add a tag
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
    
    /// Set time in force
    pub fn with_time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }
    
    /// Convert to core order side
    pub fn to_core_side(&self) -> Side {
        match self.side {
            OrderSide::Buy | OrderSide::BuyCover => Side::Bid,
            OrderSide::Sell | OrderSide::SellShort => Side::Ask,
        }
    }
}

/// Order update request
#[derive(Debug, Clone)]
pub struct OrderUpdate {
    /// Order to update
    pub order_id: OrderId,
    /// New quantity (None to keep current)
    pub new_quantity: Option<Quantity>,
    /// New price (None to keep current)
    pub new_price: Option<Price>,
}

/// Strategy metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct StrategyMetrics {
    /// Custom metrics as key-value pairs
    pub values: HashMap<String, f64>,
    /// Timestamp of metrics
    pub timestamp: u64,
}

impl StrategyMetrics {
    /// Create new metrics
    pub fn new(timestamp: u64) -> Self {
        Self {
            values: HashMap::new(),
            timestamp,
        }
    }
    
    /// Add a metric
    pub fn add(&mut self, key: impl Into<String>, value: f64) {
        self.values.insert(key.into(), value);
    }
    
    /// Get a metric
    pub fn get(&self, key: &str) -> Option<f64> {
        self.values.get(key).copied()
    }
}

/// Output from strategy processing
#[derive(Debug, Clone, Default)]
pub struct StrategyOutput {
    /// New orders to submit
    pub orders: Vec<OrderRequest>,
    /// Orders to cancel
    pub cancellations: Vec<OrderId>,
    /// Orders to update
    pub updates: Vec<OrderUpdate>,
    /// Metrics update
    pub metrics: Option<StrategyMetrics>,
}

impl StrategyOutput {
    /// Create empty output
    pub fn none() -> Self {
        Self::default()
    }
    
    /// Create output with a single order
    pub fn with_order(order: OrderRequest) -> Self {
        Self {
            orders: vec![order],
            ..Default::default()
        }
    }
    
    /// Create output with multiple orders
    pub fn with_orders(orders: Vec<OrderRequest>) -> Self {
        Self {
            orders,
            ..Default::default()
        }
    }
    
    /// Create output to cancel orders
    pub fn cancel_orders(order_ids: Vec<OrderId>) -> Self {
        Self {
            cancellations: order_ids,
            ..Default::default()
        }
    }
    
    /// Add an order
    pub fn add_order(&mut self, order: OrderRequest) {
        self.orders.push(order);
    }
    
    /// Add a cancellation
    pub fn add_cancellation(&mut self, order_id: OrderId) {
        self.cancellations.push(order_id);
    }
    
    /// Add an update
    pub fn add_update(&mut self, update: OrderUpdate) {
        self.updates.push(update);
    }
    
    /// Set metrics
    pub fn set_metrics(&mut self, metrics: StrategyMetrics) {
        self.metrics = Some(metrics);
    }
    
    /// Check if output is empty
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty() && 
        self.cancellations.is_empty() && 
        self.updates.is_empty() &&
        self.metrics.is_none()
    }
}