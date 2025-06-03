//! Main backtesting engine
//!
//! This module orchestrates the backtesting process, managing event flow,
//! strategy execution, and performance tracking.

use crate::core::types::{InstrumentId, Price, Quantity};
use crate::core::traits::MarketDataSource;
use crate::market_data::events::MarketEvent;
use crate::market_data::reader::FileReader;
use crate::strategy::{Strategy, StrategyContext, StrategyOutput};
use crate::features::{FeatureExtractor, FeatureConfig, FeaturePosition, RiskLimits};
use crate::backtest::events::{BacktestEvent, TimerEvent, OrderUpdateEvent, FillEvent, EventPriority};
use crate::backtest::execution::{ExecutionEngine, LatencyModel, FillModel};
use crate::backtest::market_state::MarketStateManager;
use crate::backtest::position::{PositionManager, PositionStats, PortfolioStats};
use crate::backtest::metrics::{MetricsCollector, PerformanceMetrics};
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

/// Event with priority and timestamp for ordering
#[derive(Debug, Clone)]
struct PrioritizedEvent {
    event: BacktestEvent,
    timestamp: u64,
    priority: EventPriority,
}

impl PartialEq for PrioritizedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp && self.priority == other.priority
    }
}

impl Eq for PrioritizedEvent {}

impl PartialOrd for PrioritizedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // First by timestamp (earlier is greater priority)
        match self.timestamp.cmp(&other.timestamp) {
            Ordering::Equal => {
                // Then by priority (lower enum value is higher priority)
                other.priority.cmp(&self.priority)
            }
            other => other.reverse(),
        }
    }
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Start timestamp (microseconds)
    pub start_time: Option<u64>,
    /// End timestamp (microseconds)
    pub end_time: Option<u64>,
    /// Latency model for execution
    pub latency_model: LatencyModel,
    /// Fill model for execution
    pub fill_model: FillModel,
    /// Commission per contract
    pub commission_per_contract: f64,
    /// Initial capital
    pub initial_capital: f64,
    /// Whether to calculate features
    pub calculate_features: bool,
    /// Feature configuration if calculating
    pub feature_config: Option<FeatureConfig>,
    /// Maximum events to process (for testing)
    pub max_events: Option<usize>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            start_time: None,
            end_time: None,
            latency_model: LatencyModel::Fixed(100), // 100 microseconds
            fill_model: FillModel::Realistic {
                maker_fill_prob: 0.7,
                taker_slippage_ticks: 1,
            },
            commission_per_contract: 0.5,
            initial_capital: 100_000.0,
            calculate_features: true,
            feature_config: None,
            max_events: None,
        }
    }
}

/// Strategy wrapper with state
struct StrategyWrapper {
    strategy: Box<dyn Strategy>,
    context: StrategyContext,
    feature_extractor: Option<FeatureExtractor>,
    capital: f64,
    position: HashMap<InstrumentId, i64>,
    next_timer: Option<u64>,
}

/// Main backtesting engine
pub struct BacktestEngine {
    /// Configuration
    config: BacktestConfig,
    /// Event queue
    event_queue: BinaryHeap<PrioritizedEvent>,
    /// Strategies
    strategies: HashMap<String, StrategyWrapper>,
    /// Market state manager
    market_state: Arc<RwLock<MarketStateManager>>,
    /// Execution engine
    execution_engine: ExecutionEngine,
    /// Position manager
    position_manager: PositionManager,
    /// Metrics collector
    metrics_collector: MetricsCollector,
    /// Current time
    current_time: u64,
    /// Event count for debugging
    events_processed: usize,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        let market_state = Arc::new(RwLock::new(MarketStateManager::new()));
        let execution_engine = ExecutionEngine::new(
            config.latency_model.clone(),
            config.fill_model.clone(),
            market_state.clone(),
        );
        
        // Create position manager with global risk limits
        let global_risk_limits = RiskLimits::default();
        let position_manager = PositionManager::new(global_risk_limits);
        
        // Create metrics collector
        let metrics_collector = MetricsCollector::new(config.initial_capital);
        
        Self {
            config,
            event_queue: BinaryHeap::new(),
            strategies: HashMap::new(),
            market_state,
            execution_engine,
            position_manager,
            metrics_collector,
            current_time: 0,
            events_processed: 0,
        }
    }
    
    /// Add a strategy to the backtest
    pub fn add_strategy(&mut self, mut strategy: Box<dyn Strategy>) -> Result<(), String> {
        let config = strategy.config();
        let strategy_id = config.id.clone();
        let uses_timer = config.uses_timer;
        let timer_interval_us = config.timer_interval_us;
        let max_position_size = config.max_position_size;
        let daily_max_loss = config.daily_max_loss;
        let max_loss = config.max_loss;
        let max_orders_per_minute = config.max_orders_per_minute;
        
        // Create feature extractor if needed
        let feature_extractor = if self.config.calculate_features {
            let feature_config = self.config.feature_config.clone()
                .unwrap_or_default();
            Some(FeatureExtractor::new(feature_config))
        } else {
            None
        };
        
        // Create initial context
        let risk_limits = RiskLimits {
            max_position: max_position_size,
            max_order_size: max_position_size / 2,
            max_loss,
            daily_max_loss,
            max_orders_per_minute,
        };
        
        let position = FeaturePosition::default();
        
        let context = StrategyContext::new(
            strategy_id.clone(),
            self.current_time,
            position,
            risk_limits.clone(),
            true, // is_backtesting
        );
        
        // Initialize strategy
        strategy.initialize(&context)?;
        
        // Calculate initial timer if needed
        let next_timer = if uses_timer {
            timer_interval_us.map(|interval| self.current_time + interval)
        } else {
            None
        };
        
        // Create wrapper
        let wrapper = StrategyWrapper {
            strategy,
            context,
            feature_extractor,
            capital: self.config.initial_capital,
            position: HashMap::new(),
            next_timer,
        };
        
        // Register strategy with position manager
        self.position_manager.add_strategy(strategy_id.clone(), risk_limits);
        
        self.strategies.insert(strategy_id, wrapper);
        Ok(())
    }
    
    /// Run backtest on market data files
    pub fn run<P: AsRef<Path>>(&mut self, data_files: &[P]) -> Result<EngineReport, String> {
        // Load initial market data
        for file_path in data_files {
            self.load_market_data(file_path)?;
        }
        
        // Process events until done
        while let Some(event) = self.next_event() {
            self.process_event(event)?;
            
            // Check max events limit
            if let Some(max) = self.config.max_events {
                if self.events_processed >= max {
                    break;
                }
            }
        }
        
        // Generate report
        Ok(self.generate_report())
    }
    
    /// Load market data from file
    fn load_market_data<P: AsRef<Path>>(&mut self, file_path: P) -> Result<(), String> {
        let paths = vec![PathBuf::from(file_path.as_ref())];
        let mut reader = FileReader::new(paths)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        
        // Read events and add to queue
        let mut count = 0;
        while let Some(update) = reader.next_update() {
            // Convert to market event
            let event = self.market_update_to_event(update)?;
            
            // Add to queue
            self.add_event(BacktestEvent::Market(event));
            count += 1;
        }
        
        println!("Loaded {} events from {:?}", count, file_path.as_ref());
        Ok(())
    }
    
    /// Convert market update to market event (simplified)
    fn market_update_to_event(&self, update: crate::core::MarketUpdate) -> Result<MarketEvent, String> {
        use crate::market_data::events::TradeEvent;
        use crate::core::Side;
        
        match update {
            crate::core::MarketUpdate::Trade(trade) => {
                Ok(MarketEvent::Trade(TradeEvent {
                    instrument_id: trade.instrument_id,
                    trade_id: 0, // Not available in Trade struct
                    price: trade.price,
                    quantity: trade.quantity,
                    aggressor_side: trade.side,
                    timestamp: trade.timestamp,
                    buyer_order_id: None,
                    seller_order_id: None,
                }))
            }
            crate::core::MarketUpdate::OrderBook(_) => {
                // Convert to trade event for now
                Ok(MarketEvent::Trade(TradeEvent {
                    instrument_id: 1,
                    trade_id: 1,
                    price: Price::from(100i64),
                    quantity: Quantity::from(1u32),
                    aggressor_side: Side::Bid,
                    timestamp: self.current_time,
                    buyer_order_id: None,
                    seller_order_id: None,
                }))
            }
        }
    }
    
    /// Add event to queue
    fn add_event(&mut self, event: BacktestEvent) {
        if let Some(timestamp) = event.timestamp() {
            let priority = event.priority();
            self.event_queue.push(PrioritizedEvent {
                event,
                timestamp,
                priority,
            });
        }
    }
    
    /// Get next event from queue
    fn next_event(&mut self) -> Option<BacktestEvent> {
        // First process any pending fills
        let fills = self.execution_engine.process_orders(self.current_time);
        for fill in fills {
            self.add_event(BacktestEvent::Fill(fill));
        }
        
        // Check for timer events
        self.check_timer_events();
        
        // Check for daily reset
        self.check_daily_reset();
        
        // Get next event
        if let Some(prioritized) = self.event_queue.pop() {
            self.current_time = prioritized.timestamp;
            Some(prioritized.event)
        } else {
            None
        }
    }
    
    /// Check for daily reset
    fn check_daily_reset(&mut self) {
        // Check if we've crossed a day boundary (simplified - just check 24 hours)
        const DAY_MICROSECONDS: u64 = 24 * 60 * 60 * 1_000_000;
        let current_day = self.current_time / DAY_MICROSECONDS;
        let last_reset_day = self.metrics_collector.last_daily_reset / DAY_MICROSECONDS;
        
        if current_day > last_reset_day {
            self.metrics_collector.reset_daily(current_day * DAY_MICROSECONDS);
            self.position_manager.reset_daily_pnl(current_day * DAY_MICROSECONDS);
        }
    }
    
    /// Check and add timer events
    fn check_timer_events(&mut self) {
        let mut timer_events = Vec::new();
        
        for (strategy_id, wrapper) in &self.strategies {
            if let Some(next_timer) = wrapper.next_timer {
                if next_timer <= self.current_time {
                    timer_events.push(TimerEvent {
                        strategy_id: strategy_id.clone(),
                        timestamp: next_timer,
                    });
                }
            }
        }
        
        for event in timer_events {
            self.add_event(BacktestEvent::Timer(event));
        }
    }
    
    /// Process a single event
    fn process_event(&mut self, event: BacktestEvent) -> Result<(), String> {
        self.events_processed += 1;
        
        match event {
            BacktestEvent::Market(market_event) => {
                self.process_market_event(market_event)?;
            }
            BacktestEvent::Timer(timer_event) => {
                self.process_timer_event(timer_event)?;
            }
            BacktestEvent::OrderUpdate(update) => {
                self.process_order_update(update)?;
            }
            BacktestEvent::Fill(fill) => {
                self.process_fill(fill)?;
            }
            BacktestEvent::EndOfData => {
                // Nothing to do
            }
        }
        
        Ok(())
    }
    
    /// Process market event
    fn process_market_event(&mut self, event: MarketEvent) -> Result<(), String> {
        // Update market state
        {
            let mut market_state = self.market_state.write().unwrap();
            market_state.process_event(&event);
        }
        
        // Update position manager with market prices
        self.update_position_prices(&event);
        
        // Collect strategy outputs first to avoid borrowing conflicts
        let mut strategy_outputs = Vec::new();
        
        // Update features and dispatch to strategies
        for (_strategy_id, wrapper) in &mut self.strategies {
            // Update feature extractor
            if let Some(ref mut _extractor) = wrapper.feature_extractor {
                // TODO: Convert market event to order book event for features
                // extractor.on_event(&order_book_event);
            }
            
            // Update context with market state
            wrapper.context.current_time = self.current_time;
            
            // Call strategy
            let output = wrapper.strategy.on_market_event(&event, &wrapper.context);
            strategy_outputs.push((wrapper.context.strategy_id.clone(), output));
        }
        
        // Process outputs
        for (strategy_id, output) in strategy_outputs {
            self.process_strategy_output(&strategy_id, output)?;
        }
        
        Ok(())
    }
    
    /// Process timer event
    fn process_timer_event(&mut self, event: TimerEvent) -> Result<(), String> {
        let output = if let Some(wrapper) = self.strategies.get_mut(&event.strategy_id) {
            // Update context
            wrapper.context.current_time = event.timestamp;
            
            // Call strategy
            let output = wrapper.strategy.on_timer(event.timestamp, &wrapper.context);
            
            // Schedule next timer
            if let Some(interval) = wrapper.strategy.config().timer_interval_us {
                wrapper.next_timer = Some(event.timestamp + interval);
            }
            
            Some(output)
        } else {
            None
        };
        
        // Process output if we have one
        if let Some(output) = output {
            self.process_strategy_output(&event.strategy_id, output)?;
        }
        
        Ok(())
    }
    
    /// Process order update
    fn process_order_update(&mut self, update: OrderUpdateEvent) -> Result<(), String> {
        if let Some(wrapper) = self.strategies.get_mut(&update.strategy_id) {
            // Update pending orders in context
            // TODO: Implement order tracking in context
            
            // Notify strategy if rejected
            if update.status == crate::backtest::events::OrderStatus::Rejected {
                wrapper.strategy.on_order_rejected(
                    update.order_id,
                    update.message.unwrap_or_default(),
                    &wrapper.context,
                );
            }
        }
        
        Ok(())
    }
    
    /// Process fill event
    fn process_fill(&mut self, fill: FillEvent) -> Result<(), String> {
        // Apply fill to position manager first
        self.position_manager.apply_fill(&fill)?;
        
        // Update metrics collector
        self.metrics_collector.process_fill(&fill);
        
        if let Some(wrapper) = self.strategies.get_mut(&fill.strategy_id) {
            // Update position
            let position = wrapper.position.entry(fill.instrument_id).or_insert(0);
            match fill.side {
                crate::core::Side::Bid => *position += fill.quantity.as_i64(),
                crate::core::Side::Ask => *position -= fill.quantity.as_i64(),
            }
            
            // Update P&L
            let cost = fill.total_cost();
            wrapper.capital -= cost;
            
            // Update context position
            wrapper.context.position.quantity = *position;
            
            // Notify strategy
            wrapper.strategy.on_fill(
                fill.price,
                fill.quantity.as_i64(),
                fill.timestamp,
                &wrapper.context,
            );
        }
        
        Ok(())
    }
    
    /// Process strategy output
    fn process_strategy_output(&mut self, strategy_id: &str, output: StrategyOutput) -> Result<(), String> {
        // Process order requests
        for order in output.orders {
            let order_id = self.execution_engine.submit_order(
                order,
                strategy_id.to_string(),
                self.current_time,
            );
            
            // Add order update event
            self.add_event(BacktestEvent::OrderUpdate(OrderUpdateEvent {
                order_id,
                strategy_id: strategy_id.to_string(),
                status: crate::backtest::events::OrderStatus::Accepted,
                timestamp: self.current_time,
                message: None,
            }));
        }
        
        // Process order cancellations
        for cancel in output.updates {
            if let Some(update) = self.execution_engine.cancel_order(cancel.order_id, self.current_time) {
                self.add_event(BacktestEvent::OrderUpdate(update));
            }
        }
        
        Ok(())
    }
    
    /// Update position manager with current market prices
    fn update_position_prices(&mut self, event: &MarketEvent) {
        // Extract price updates from market events
        let mut price_updates = HashMap::new();
        
        match event {
            crate::market_data::events::MarketEvent::Trade(trade) => {
                price_updates.insert(trade.instrument_id, trade.price);
            }
            crate::market_data::events::MarketEvent::BBO(bbo) => {
                if let (Some(bid), Some(ask)) = (bbo.bid_price, bbo.ask_price) {
                    let mid_price = Price::from_f64((bid.as_f64() + ask.as_f64()) / 2.0);
                    price_updates.insert(bbo.instrument_id, mid_price);
                }
            }
            _ => {} // Other events don't provide price updates
        }
        
        if !price_updates.is_empty() {
            self.position_manager.update_market_prices(price_updates, self.current_time);
        }
    }
    
    /// Generate backtest report
    fn generate_report(&self) -> EngineReport {
        let mut strategy_results = Vec::new();
        
        for (strategy_id, wrapper) in &self.strategies {
            let final_capital = wrapper.capital;
            let total_pnl = final_capital - self.config.initial_capital;
            let positions: Vec<_> = wrapper.position.iter()
                .map(|(k, v)| (*k, *v))
                .collect();
            
            // Get position stats from position manager
            let position_stats = self.position_manager
                .get_strategy_tracker(strategy_id)
                .map(|tracker| tracker.get_stats());
            
            strategy_results.push(StrategyResult {
                strategy_id: strategy_id.clone(),
                initial_capital: self.config.initial_capital,
                final_capital,
                total_pnl,
                final_positions: positions,
                position_stats,
            });
        }
        
        // Get portfolio-wide statistics
        let portfolio_stats = self.position_manager.get_portfolio_stats();
        
        // Get performance metrics
        let performance_metrics = self.metrics_collector.calculate_metrics();
        let trades = self.metrics_collector.get_trades().to_vec();
        let equity_curve = self.metrics_collector.get_equity_curve().to_vec();
        
        EngineReport {
            config: self.config.clone(),
            events_processed: self.events_processed,
            strategy_results,
            portfolio_stats,
            performance_metrics,
            trades,
            equity_curve,
        }
    }
}

/// Engine backtest report
#[derive(Debug, Clone)]
pub struct EngineReport {
    /// Configuration used
    pub config: BacktestConfig,
    /// Total events processed
    pub events_processed: usize,
    /// Results by strategy
    pub strategy_results: Vec<StrategyResult>,
    /// Portfolio-wide statistics
    pub portfolio_stats: PortfolioStats,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Trades
    pub trades: Vec<crate::backtest::metrics::Trade>,
    /// Equity curve
    pub equity_curve: Vec<crate::backtest::metrics::EquityPoint>,
}

/// Results for a single strategy
#[derive(Debug, Clone)]
pub struct StrategyResult {
    /// Strategy ID
    pub strategy_id: String,
    /// Initial capital
    pub initial_capital: f64,
    /// Final capital
    pub final_capital: f64,
    /// Total P&L
    pub total_pnl: f64,
    /// Final positions
    pub final_positions: Vec<(InstrumentId, i64)>,
    /// Position statistics from position manager
    pub position_stats: Option<PositionStats>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_event_ordering() {
        let mut heap = BinaryHeap::new();
        
        // Add events with different timestamps and priorities
        heap.push(PrioritizedEvent {
            event: BacktestEvent::EndOfData,
            timestamp: 1000,
            priority: EventPriority::Timer,
        });
        
        heap.push(PrioritizedEvent {
            event: BacktestEvent::EndOfData,
            timestamp: 1000,
            priority: EventPriority::MarketData,
        });
        
        heap.push(PrioritizedEvent {
            event: BacktestEvent::EndOfData,
            timestamp: 900,
            priority: EventPriority::Timer,
        });
        
        // Should get events in order: 900 first, then 1000 with MarketData priority
        let first = heap.pop().unwrap();
        assert_eq!(first.timestamp, 900);
        
        let second = heap.pop().unwrap();
        assert_eq!(second.timestamp, 1000);
        assert_eq!(second.priority, EventPriority::MarketData);
        
        let third = heap.pop().unwrap();
        assert_eq!(third.timestamp, 1000);
        assert_eq!(third.priority, EventPriority::Timer);
    }
    
    #[test]
    fn test_backtest_engine_creation() {
        let config = BacktestConfig::default();
        let engine = BacktestEngine::new(config);
        
        assert_eq!(engine.events_processed, 0);
        assert_eq!(engine.strategies.len(), 0);
    }
}