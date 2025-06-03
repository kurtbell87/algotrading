//! Core strategy trait definitions

use crate::core::types::{InstrumentId, Price};
use crate::market_data::events::MarketEvent;
use crate::strategy::{StrategyId, StrategyError, StrategyState};
use crate::strategy::context::StrategyContext;
use crate::strategy::output::StrategyOutput;
use std::collections::HashMap;

/// Core trait that all trading strategies must implement
pub trait Strategy: Send + Sync {
    /// Called when strategy is initialized
    fn initialize(&mut self, _context: &StrategyContext) -> Result<(), StrategyError> {
        Ok(())
    }
    
    /// Called on each market event
    fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput;
    
    /// Called periodically for time-based strategies
    fn on_timer(&mut self, _timestamp: u64, _context: &StrategyContext) -> StrategyOutput {
        StrategyOutput::default()
    }
    
    /// Called when an order is filled
    fn on_fill(&mut self, _fill_price: Price, _fill_quantity: i64, _timestamp: u64, _context: &StrategyContext) {
        // Default implementation does nothing
    }
    
    /// Called when an order is rejected
    fn on_order_rejected(&mut self, _order_id: u64, _reason: String, _context: &StrategyContext) {
        // Default implementation does nothing
    }
    
    /// Called at end of trading session
    fn on_session_end(&mut self, _timestamp: u64, _context: &StrategyContext) -> StrategyOutput {
        // Default: close all positions
        StrategyOutput::default()
    }
    
    /// Get strategy configuration
    fn config(&self) -> &StrategyConfig;
    
    /// Get current strategy state
    fn state(&self) -> StrategyState {
        StrategyState::Active
    }
    
    /// Get strategy metrics for monitoring
    fn metrics(&self) -> Option<HashMap<String, f64>> {
        None
    }
}

/// Strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Strategy identifier
    pub id: StrategyId,
    /// Strategy name for display
    pub name: String,
    /// Instruments this strategy trades
    pub instruments: Vec<InstrumentId>,
    /// Maximum position size per instrument
    pub max_position_size: i64,
    /// Maximum number of orders per minute
    pub max_orders_per_minute: u32,
    /// Risk limits
    pub max_loss: f64,
    pub daily_max_loss: f64,
    /// Whether strategy uses timer events
    pub uses_timer: bool,
    /// Timer interval in microseconds (if uses_timer)
    pub timer_interval_us: Option<u64>,
    /// Custom parameters
    pub params: HashMap<String, String>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            id: "default".to_string(),
            name: "Default Strategy".to_string(),
            instruments: vec![],
            max_position_size: 100,
            max_orders_per_minute: 100,
            max_loss: 10000.0,
            daily_max_loss: 5000.0,
            uses_timer: false,
            timer_interval_us: None,
            params: HashMap::new(),
        }
    }
}

impl StrategyConfig {
    /// Create a new strategy configuration
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            ..Default::default()
        }
    }
    
    /// Add an instrument to trade
    pub fn with_instrument(mut self, instrument_id: InstrumentId) -> Self {
        self.instruments.push(instrument_id);
        self
    }
    
    /// Set max position size
    pub fn with_max_position(mut self, size: i64) -> Self {
        self.max_position_size = size;
        self
    }
    
    /// Enable timer with interval
    pub fn with_timer(mut self, interval_us: u64) -> Self {
        self.uses_timer = true;
        self.timer_interval_us = Some(interval_us);
        self
    }
    
    /// Add a parameter
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
    
    /// Get a parameter value
    pub fn get_param(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(|s| s.as_str())
    }
    
    /// Get a parameter as f64
    pub fn get_param_f64(&self, key: &str) -> Option<f64> {
        self.get_param(key)?.parse().ok()
    }
    
    /// Get a parameter as i64
    pub fn get_param_i64(&self, key: &str) -> Option<i64> {
        self.get_param(key)?.parse().ok()
    }
    
    /// Get a parameter as bool
    pub fn get_param_bool(&self, key: &str) -> Option<bool> {
        self.get_param(key)?.parse().ok()
    }
}