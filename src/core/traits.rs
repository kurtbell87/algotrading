use crate::core::types::*;
use std::error::Error;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Market data source trait
pub trait MarketDataSource {
    fn subscribe(&mut self, instruments: Vec<InstrumentId>) -> Result<()>;
    fn next_update(&mut self) -> Option<MarketUpdate>;
}

/// Order book trait
pub trait OrderBook {
    fn best_bid(&self) -> Option<PriceLevel>;
    fn best_ask(&self) -> Option<PriceLevel>;
    fn depth(&self, levels: usize) -> BookDepth;
    fn spread(&self) -> Option<Price>;
}

/// Feature calculator trait
pub trait FeatureCalculator: Send + Sync {
    type Input;
    type Output;

    fn calculate(&mut self, input: &Self::Input) -> Self::Output;
    fn window_size(&self) -> usize;
    fn name(&self) -> &str;
}

/// Feature engine trait
pub trait FeatureEngine {
    fn calculate_features(&mut self, book: &dyn OrderBook) -> FeatureVector;
    fn get_feature_names(&self) -> Vec<String>;
}

/// Feature vector
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub values: Vec<f64>,
    pub names: Vec<String>,
    pub timestamp: u64,
}

/// Strategy execution context
pub struct Context<'a> {
    pub executor: &'a mut dyn Executor,
    pub risk_manager: &'a dyn RiskManager,
    pub position_tracker: &'a dyn PositionTracker,
}

/// Trading strategy trait
pub trait Strategy {
    fn on_book_update(&mut self, ctx: &mut Context, book: &dyn OrderBook);
    fn on_features(&mut self, ctx: &mut Context, features: &FeatureVector);
    fn on_trade(&mut self, ctx: &mut Context, trade: &Trade);
    fn on_fill(&mut self, ctx: &mut Context, fill: &Fill);
}

/// ML model trait
pub trait MLModel {
    fn predict(&self, features: &FeatureVector) -> Prediction;
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub signal: f64,
    pub confidence: f64,
}

/// Order executor trait
pub trait Executor {
    fn submit_order(&mut self, order: Order) -> Result<OrderId>;
    fn cancel_order(&mut self, id: OrderId) -> Result<()>;
    fn modify_order(&mut self, id: OrderId, new_order: Order) -> Result<()>;
}

/// Risk manager trait
pub trait RiskManager {
    fn check_order(&self, order: &Order, position: &Position) -> Result<()>;
    fn get_limits(&self) -> &RiskLimits;
}

/// Position tracker trait  
pub trait PositionTracker {
    fn get_position(&self, instrument_id: InstrumentId) -> Option<&Position>;
    fn update_position(&mut self, fill: &Fill);
    fn get_all_positions(&self) -> Vec<&Position>;
}
