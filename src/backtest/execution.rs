//! Order execution engine for backtesting
//!
//! This module handles order matching, fill generation, and execution simulation
//! including realistic latency and market impact models.

use crate::backtest::events::{FillEvent, OrderStatus, OrderUpdateEvent};
use crate::backtest::market_state::MarketStateManager;
use crate::core::Side;
use crate::core::types::{InstrumentId, OrderId, Price, Quantity};
use crate::order_book::book::Book;
use crate::strategy::OrderSide;
use crate::strategy::{OrderRequest, TimeInForce};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

/// Convert OrderSide to Side for fill events
fn convert_order_side_to_side(side: OrderSide) -> Side {
    match side {
        OrderSide::Buy | OrderSide::BuyCover => Side::Bid,
        OrderSide::Sell | OrderSide::SellShort => Side::Ask,
    }
}

/// Latency model for order execution
#[derive(Debug, Clone)]
pub enum LatencyModel {
    /// Fixed latency in microseconds
    Fixed(u64),
    /// Variable latency with mean and std deviation
    Normal { mean: u64, std_dev: u64 },
    /// Latency that varies by order size
    SizeDependent { base: u64, per_unit: f64 },
    /// No latency (for testing)
    Zero,
}

impl LatencyModel {
    /// Calculate latency for an order
    pub fn calculate(&self, quantity: &Quantity) -> u64 {
        match self {
            Self::Fixed(latency) => *latency,
            Self::Normal { mean, std_dev } => {
                // Simple approximation - in production use proper random
                let variation = (*std_dev as f64 * 0.5) as u64;
                mean.saturating_add(variation)
            }
            Self::SizeDependent { base, per_unit } => {
                base + (quantity.as_u32() as f64 * per_unit) as u64
            }
            Self::Zero => 0,
        }
    }
}

/// Fill model determines how orders are filled
#[derive(Clone)]
pub enum FillModel {
    /// Always fill at best price if available
    Optimistic,
    /// Fill at mid-market price
    MidPoint,
    /// Realistic fill considering queue position
    Realistic {
        /// Probability of getting filled at touch
        maker_fill_prob: f64,
        /// Slippage in ticks for aggressive orders
        taker_slippage_ticks: i64,
    },
    /// Custom fill logic
    Custom(Box<dyn FillLogic>),
}

impl std::fmt::Debug for FillModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Optimistic => write!(f, "Optimistic"),
            Self::MidPoint => write!(f, "MidPoint"),
            Self::Realistic {
                maker_fill_prob,
                taker_slippage_ticks,
            } => f
                .debug_struct("Realistic")
                .field("maker_fill_prob", maker_fill_prob)
                .field("taker_slippage_ticks", taker_slippage_ticks)
                .finish(),
            Self::Custom(_) => write!(f, "Custom(..)"),
        }
    }
}

/// Trait for custom fill logic
pub trait FillLogic: Send + Sync {
    /// Determine fill price and quantity
    fn calculate_fill(
        &self,
        order: &OrderRequest,
        book: &Book,
        timestamp: u64,
    ) -> Option<(Price, Quantity)>;

    /// Clone the fill logic
    fn clone_box(&self) -> Box<dyn FillLogic>;
}

impl Clone for Box<dyn FillLogic> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Pending order in the execution queue
#[derive(Debug, Clone)]
pub struct PendingOrder {
    /// Order details
    order: OrderRequest,
    /// Strategy that placed the order
    strategy_id: String,
    /// Time when order becomes active (after latency)
    active_time: u64,
    /// Order status
    status: OrderStatus,
    /// Filled quantity so far
    filled_quantity: Quantity,
    /// Average fill price
    avg_fill_price: Option<Price>,
}

/// Execution engine handles order matching and fills
pub struct ExecutionEngine {
    /// Latency model
    latency_model: LatencyModel,
    /// Fill model
    fill_model: FillModel,
    /// Pending orders by order ID
    pending_orders: HashMap<OrderId, PendingOrder>,
    /// Order queue by instrument and time
    order_queue: HashMap<InstrumentId, VecDeque<OrderId>>,
    /// Next order ID
    next_order_id: OrderId,
    /// Reference to market state
    market_state: Arc<RwLock<MarketStateManager>>,
    /// Commission model
    commission_per_contract: f64,
}

impl ExecutionEngine {
    /// Create a new execution engine
    pub fn new(
        latency_model: LatencyModel,
        fill_model: FillModel,
        market_state: Arc<RwLock<MarketStateManager>>,
    ) -> Self {
        Self {
            latency_model,
            fill_model,
            pending_orders: HashMap::new(),
            order_queue: HashMap::new(),
            next_order_id: 1,
            market_state,
            commission_per_contract: 0.5, // Default $0.50 per contract
        }
    }

    /// Set commission per contract
    pub fn set_commission(&mut self, commission: f64) {
        self.commission_per_contract = commission;
    }

    /// Submit an order
    pub fn submit_order(
        &mut self,
        order: OrderRequest,
        strategy_id: String,
        current_time: u64,
    ) -> OrderId {
        let order_id = self.next_order_id;
        self.next_order_id += 1;

        // Calculate when order becomes active
        let latency = self.latency_model.calculate(&order.quantity);
        let active_time = current_time + latency;

        // Create pending order
        let pending = PendingOrder {
            order,
            strategy_id,
            active_time,
            status: OrderStatus::Accepted,
            filled_quantity: Quantity::new(0),
            avg_fill_price: None,
        };

        // Add to pending orders and queue
        self.pending_orders.insert(order_id, pending.clone());
        self.order_queue
            .entry(pending.order.instrument_id)
            .or_insert_with(VecDeque::new)
            .push_back(order_id);

        order_id
    }

    /// Cancel an order
    pub fn cancel_order(
        &mut self,
        order_id: OrderId,
        current_time: u64,
    ) -> Option<OrderUpdateEvent> {
        if let Some(mut order) = self.pending_orders.remove(&order_id) {
            order.status = OrderStatus::Cancelled;

            // Remove from queue
            if let Some(queue) = self.order_queue.get_mut(&order.order.instrument_id) {
                queue.retain(|&id| id != order_id);
            }

            Some(OrderUpdateEvent {
                order_id,
                strategy_id: order.strategy_id,
                status: OrderStatus::Cancelled,
                timestamp: current_time,
                message: Some("Order cancelled".to_string()),
            })
        } else {
            None
        }
    }

    /// Process orders at current time
    pub fn process_orders(&mut self, current_time: u64) -> Vec<FillEvent> {
        let mut fills = Vec::new();
        let market_state = self.market_state.read().unwrap();

        // Process each instrument's order queue
        for (instrument_id, queue) in &mut self.order_queue {
            let mut completed_orders = Vec::new();

            // Get current order book
            let book_opt = market_state.get_order_book(*instrument_id);
            if book_opt.is_none() {
                continue;
            }

            let book = book_opt.unwrap();
            let book = book.read().unwrap();

            // Process orders that are now active
            for &order_id in queue.iter() {
                if let Some(pending) = self.pending_orders.get_mut(&order_id) {
                    // Skip if not yet active
                    if pending.active_time > current_time {
                        continue;
                    }

                    // Skip if already filled or cancelled
                    if pending.status != OrderStatus::Accepted
                        && pending.status != OrderStatus::PartiallyFilled
                    {
                        completed_orders.push(order_id);
                        continue;
                    }

                    // Try to fill the order
                    if let Some(fill) = Self::try_fill_order_static(
                        &self.fill_model,
                        self.commission_per_contract,
                        pending,
                        &book,
                        order_id,
                        current_time,
                    ) {
                        fills.push(fill);

                        // Check if order is complete
                        if pending.filled_quantity.as_u32() >= pending.order.quantity.as_u32() {
                            pending.status = OrderStatus::Filled;
                            completed_orders.push(order_id);
                        } else {
                            pending.status = OrderStatus::PartiallyFilled;
                        }
                    }

                    // Check time in force
                    if Self::should_expire_order(pending, current_time) {
                        pending.status = OrderStatus::Expired;
                        completed_orders.push(order_id);
                    }
                }
            }

            // Remove completed orders from queue
            for order_id in completed_orders {
                queue.retain(|&id| id != order_id);
            }
        }

        fills
    }

    /// Try to fill an order against the current book
    fn try_fill_order_static(
        fill_model: &FillModel,
        commission_per_contract: f64,
        pending: &mut PendingOrder,
        book: &Book,
        order_id: OrderId,
        timestamp: u64,
    ) -> Option<FillEvent> {
        let remaining_qty = Quantity::new(
            pending
                .order
                .quantity
                .as_u32()
                .saturating_sub(pending.filled_quantity.as_u32()),
        );
        if remaining_qty.as_u32() == 0 {
            return None;
        }

        // Get fill price and quantity based on fill model
        let (fill_price, fill_quantity) = match fill_model {
            FillModel::Optimistic => Self::optimistic_fill(pending, book, remaining_qty)?,
            FillModel::MidPoint => Self::midpoint_fill(pending, book, remaining_qty)?,
            FillModel::Realistic {
                maker_fill_prob,
                taker_slippage_ticks,
            } => Self::realistic_fill(
                pending,
                book,
                remaining_qty,
                *maker_fill_prob,
                *taker_slippage_ticks,
            )?,
            FillModel::Custom(logic) => logic.calculate_fill(&pending.order, book, timestamp)?,
        };

        // Update pending order
        pending.filled_quantity =
            Quantity::new(pending.filled_quantity.as_u32() + fill_quantity.as_u32());

        // Update average fill price
        if let Some(avg_price) = pending.avg_fill_price {
            let prev_qty = pending.filled_quantity.as_u32() - fill_quantity.as_u32();
            let total_value =
                avg_price.as_f64() * prev_qty as f64 + fill_price.as_f64() * fill_quantity.as_f64();
            pending.avg_fill_price = Some(Price::from_f64(
                total_value / pending.filled_quantity.as_f64(),
            ));
        } else {
            pending.avg_fill_price = Some(fill_price);
        }

        // Create fill event
        Some(FillEvent {
            fill_id: timestamp, // Simple fill ID
            order_id,
            strategy_id: pending.strategy_id.clone(),
            instrument_id: pending.order.instrument_id,
            price: fill_price,
            quantity: fill_quantity,
            side: convert_order_side_to_side(pending.order.side),
            timestamp,
            commission: commission_per_contract * fill_quantity.as_f64(),
            is_maker: matches!(
                pending.order.order_type,
                crate::strategy::output::OrderType::Limit
            ),
        })
    }

    /// Optimistic fill - always fill at best available price
    fn optimistic_fill(
        pending: &PendingOrder,
        book: &Book,
        remaining_qty: Quantity,
    ) -> Option<(Price, Quantity)> {
        let (best_bid, best_ask) = book.bbo();

        match pending.order.side {
            OrderSide::Buy | OrderSide::BuyCover => {
                // Buying - check ask
                let ask = best_ask?;
                let fill_qty = remaining_qty.as_u32().min(ask.size);
                Some((Price::new(ask.price), Quantity::new(fill_qty)))
            }
            OrderSide::Sell | OrderSide::SellShort => {
                // Selling - check bid
                let bid = best_bid?;
                let fill_qty = remaining_qty.as_u32().min(bid.size);
                Some((Price::new(bid.price), Quantity::new(fill_qty)))
            }
        }
    }

    /// Midpoint fill - fill at mid-market price
    fn midpoint_fill(
        _pending: &PendingOrder,
        book: &Book,
        remaining_qty: Quantity,
    ) -> Option<(Price, Quantity)> {
        let (best_bid, best_ask) = book.bbo();
        let bid = best_bid?;
        let ask = best_ask?;

        let mid_price = Price::from_f64((bid.price as f64 + ask.price as f64) / 2.0);

        // Fill entire remaining quantity at mid
        Some((mid_price, remaining_qty))
    }

    /// Realistic fill - considers queue position and market conditions
    fn realistic_fill(
        pending: &PendingOrder,
        book: &Book,
        remaining_qty: Quantity,
        maker_fill_prob: f64,
        taker_slippage_ticks: i64,
    ) -> Option<(Price, Quantity)> {
        use crate::strategy::output::OrderType;

        match pending.order.order_type {
            OrderType::Market => {
                // Market order - fill with slippage
                Self::market_order_fill(pending, book, remaining_qty, taker_slippage_ticks)
            }
            OrderType::Limit => {
                // Limit order - check if price is touched
                Self::limit_order_fill(pending, book, remaining_qty, maker_fill_prob)
            }
            _ => None, // Other order types not yet implemented
        }
    }

    /// Fill a market order with slippage
    fn market_order_fill(
        pending: &PendingOrder,
        book: &Book,
        remaining_qty: Quantity,
        slippage_ticks: i64,
    ) -> Option<(Price, Quantity)> {
        let (best_bid, best_ask) = book.bbo();

        match pending.order.side {
            OrderSide::Buy | OrderSide::BuyCover => {
                // Buying - start at ask and add slippage
                let ask = best_ask?;
                let base_price = ask.price;
                let tick_size = 25; // TODO: Get from instrument config
                let fill_price = base_price + (slippage_ticks * tick_size);
                Some((Price::new(fill_price), remaining_qty))
            }
            OrderSide::Sell | OrderSide::SellShort => {
                // Selling - start at bid and subtract slippage
                let bid = best_bid?;
                let base_price = bid.price;
                let tick_size = 25; // TODO: Get from instrument config
                let fill_price = base_price - (slippage_ticks * tick_size);
                Some((Price::new(fill_price), remaining_qty))
            }
        }
    }

    /// Fill a limit order if price is touched
    fn limit_order_fill(
        pending: &PendingOrder,
        book: &Book,
        remaining_qty: Quantity,
        maker_fill_prob: f64,
    ) -> Option<(Price, Quantity)> {
        let limit_price = pending.order.price?;
        let (best_bid, best_ask) = book.bbo();

        match pending.order.side {
            OrderSide::Buy | OrderSide::BuyCover => {
                // Buy limit - check if ask <= limit price
                let ask = best_ask?;
                if Price::new(ask.price) <= limit_price {
                    // Price touched - fill with probability
                    if Self::should_fill_maker_order(maker_fill_prob) {
                        let fill_qty = remaining_qty.as_u32().min(ask.size);
                        Some((limit_price, Quantity::new(fill_qty)))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            OrderSide::Sell | OrderSide::SellShort => {
                // Sell limit - check if bid >= limit price
                let bid = best_bid?;
                if Price::new(bid.price) >= limit_price {
                    // Price touched - fill with probability
                    if Self::should_fill_maker_order(maker_fill_prob) {
                        let fill_qty = remaining_qty.as_u32().min(bid.size);
                        Some((limit_price, Quantity::new(fill_qty)))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    /// Determine if a maker order should fill (simple probability model)
    fn should_fill_maker_order(fill_prob: f64) -> bool {
        // In production, use proper random number generator
        // For now, use a simple threshold
        fill_prob > 0.5
    }

    /// Check if order should expire based on time in force
    fn should_expire_order(pending: &PendingOrder, current_time: u64) -> bool {
        match pending.order.time_in_force {
            TimeInForce::IOC => {
                // Immediate or cancel - expire if not filled immediately
                current_time > pending.active_time && pending.filled_quantity.as_u32() == 0
            }
            TimeInForce::FOK => {
                // Fill or kill - expire if not completely filled immediately
                current_time > pending.active_time
                    && pending.filled_quantity.as_u32() < pending.order.quantity.as_u32()
            }
            TimeInForce::Day => {
                // Day order - would expire at end of day
                // TODO: Implement day boundary check
                false
            }
            TimeInForce::GTC => {
                // Good till cancelled - never expires
                false
            }
        }
    }

    /// Get all pending orders for a strategy
    pub fn get_pending_orders(&self, strategy_id: &str) -> Vec<&PendingOrder> {
        self.pending_orders
            .values()
            .filter(|order| order.strategy_id == strategy_id)
            .collect()
    }

    /// Get order status
    pub fn get_order_status(&self, order_id: OrderId) -> Option<OrderStatus> {
        self.pending_orders.get(&order_id).map(|o| o.status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::output::OrderType;
    use std::collections::HashMap;

    #[test]
    fn test_latency_models() {
        let fixed = LatencyModel::Fixed(100);
        assert_eq!(fixed.calculate(&Quantity::new(10)), 100);

        let size_dependent = LatencyModel::SizeDependent {
            base: 50,
            per_unit: 2.0,
        };
        assert_eq!(size_dependent.calculate(&Quantity::new(10)), 70);

        let zero = LatencyModel::Zero;
        assert_eq!(zero.calculate(&Quantity::new(100)), 0);
    }

    #[test]
    fn test_order_submission() {
        let market_state = Arc::new(RwLock::new(MarketStateManager::new()));
        let mut engine = ExecutionEngine::new(
            LatencyModel::Fixed(100),
            FillModel::Optimistic,
            market_state,
        );

        let order = OrderRequest {
            strategy_id: "test_strategy".to_string(),
            instrument_id: 1,
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: Quantity::new(10),
            price: Some(Price::from(100i64)),
            time_in_force: TimeInForce::GTC,
            client_order_id: None,
            tags: HashMap::new(),
        };

        let order_id = engine.submit_order(order, "test_strategy".to_string(), 1000);
        assert_eq!(order_id, 1);
        assert_eq!(engine.pending_orders.len(), 1);

        // Check order is pending
        let status = engine.get_order_status(order_id);
        assert_eq!(status, Some(OrderStatus::Accepted));
    }

    #[test]
    fn test_order_cancellation() {
        let market_state = Arc::new(RwLock::new(MarketStateManager::new()));
        let mut engine =
            ExecutionEngine::new(LatencyModel::Zero, FillModel::Optimistic, market_state);

        let order = OrderRequest {
            strategy_id: "test_strategy".to_string(),
            instrument_id: 1,
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: Quantity::new(10),
            price: Some(Price::from(100i64)),
            time_in_force: TimeInForce::GTC,
            client_order_id: None,
            tags: HashMap::new(),
        };

        let order_id = engine.submit_order(order, "test_strategy".to_string(), 1000);

        // Cancel the order
        let update = engine.cancel_order(order_id, 1100);
        assert!(update.is_some());
        assert_eq!(update.unwrap().status, OrderStatus::Cancelled);

        // Order should be removed
        assert!(engine.get_order_status(order_id).is_none());
    }
}
