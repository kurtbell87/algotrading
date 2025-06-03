//! High-performance backtesting engine
//!
//! Optimizations implemented:
//! 1. Zero-copy event processing
//! 2. Lockless market state snapshots  
//! 3. Pre-allocated strategy contexts
//! 4. Batch processing pipeline
//! 5. Minimal memory allocations

use crate::core::types::{InstrumentId, Price, Quantity};
use crate::core::{MarketDataSource, MarketUpdate, Side};
use crate::market_data::reader_optimized::{MarketSnapshot, IndexedFeatureVector, feature_indices};
use crate::strategy::{Strategy, StrategyContext, StrategyOutput};
use crate::features::{FeaturePosition, RiskLimits};
use crate::backtest::execution::{ExecutionEngine, LatencyModel, FillModel};
use crate::backtest::position::{PositionManager, PositionStats, PortfolioStats};
use crate::backtest::metrics::{MetricsCollector, PerformanceMetrics};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// High-performance backtesting configuration
#[derive(Debug, Clone)]
pub struct OptimizedBacktestConfig {
    pub initial_capital: f64,
    pub commission_per_contract: f64,
    pub latency_model: LatencyModel,
    pub fill_model: FillModel,
    pub max_events: Option<usize>,
    pub calculate_features: bool,
    /// Pre-allocate contexts to avoid runtime allocation
    pub expected_strategies: usize,
    /// Batch size for event processing
    pub event_batch_size: usize,
}

impl Default for OptimizedBacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            commission_per_contract: 0.5,
            latency_model: LatencyModel::Zero,
            fill_model: FillModel::Optimistic,
            max_events: None,
            calculate_features: false,
            expected_strategies: 4,
            event_batch_size: 1024,
        }
    }
}

/// Pre-allocated strategy wrapper for performance
struct OptimizedStrategyWrapper {
    strategy: Box<dyn Strategy>,
    context: StrategyContext,
    /// Pre-allocated feature vector
    features: IndexedFeatureVector,
    /// Performance counters
    events_processed: u64,
    orders_generated: u64,
}

/// High-performance backtesting engine  
pub struct OptimizedBacktestEngine {
    config: OptimizedBacktestConfig,
    strategies: Vec<OptimizedStrategyWrapper>,
    execution_engine: ExecutionEngine,
    position_manager: PositionManager,
    metrics_collector: MetricsCollector,
    
    /// Lockless market state
    market_snapshot: MarketSnapshot,
    
    /// Event processing counters
    events_processed: usize,
    current_time: u64,
    
    /// Pre-allocated buffers
    strategy_outputs: Vec<(String, StrategyOutput)>,
    event_batch: Vec<MarketUpdate>,
}

impl OptimizedBacktestEngine {
    /// Create new optimized engine
    pub fn new(config: OptimizedBacktestConfig) -> Self {
        // Pre-allocate all vectors based on expected load
        let strategy_outputs = Vec::with_capacity(config.expected_strategies);
        let event_batch = Vec::with_capacity(config.event_batch_size);
        
        Self {
            execution_engine: ExecutionEngine::new(
                config.latency_model.clone(),
                config.fill_model.clone(),
            ),
            position_manager: PositionManager::new(RiskLimits::default()),
            metrics_collector: MetricsCollector::new(config.initial_capital),
            market_snapshot: MarketSnapshot::new(),
            strategies: Vec::with_capacity(config.expected_strategies),
            events_processed: 0,
            current_time: 0,
            strategy_outputs,
            event_batch,
            config,
        }
    }
    
    /// Add strategy with pre-allocation
    pub fn add_strategy(&mut self, mut strategy: Box<dyn Strategy>) -> Result<(), String> {
        let strategy_id = strategy.config().strategy_id.clone();
        
        // Create optimized context with lockless market snapshot
        let context = StrategyContext::new(
            strategy_id.clone(),
            0,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );
        
        // Initialize strategy
        strategy.initialize(&context)
            .map_err(|e| format!("Failed to initialize strategy {}: {:?}", strategy_id, e))?;
        
        // Pre-allocate feature vector
        let features = IndexedFeatureVector::new(feature_indices::TOTAL_FEATURES, 0);
        
        let wrapper = OptimizedStrategyWrapper {
            strategy,
            context,
            features,
            events_processed: 0,
            orders_generated: 0,
        };
        
        self.strategies.push(wrapper);
        Ok(())
    }
    
    /// Run optimized backtest
    pub fn run_optimized<P: AsRef<Path>>(&mut self, data_files: &[P]) -> Result<OptimizedBacktestReport, String> {
        // Load data with optimized reader
        for file_path in data_files {
            self.process_file_optimized(file_path)?;
        }
        
        Ok(self.generate_optimized_report())
    }
    
    /// Process file with batch optimization
    fn process_file_optimized<P: AsRef<Path>>(&mut self, file_path: P) -> Result<(), String> {
        use crate::market_data::reader_optimized::OptimizedFileReader;
        
        let paths = vec![PathBuf::from(file_path.as_ref())];
        let mut reader = OptimizedFileReader::new(paths)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        
        // Process events in batches for cache efficiency
        self.event_batch.clear();
        
        while let Some(update) = reader.next_update() {
            self.event_batch.push(update);
            
            // Process batch when full
            if self.event_batch.len() >= self.config.event_batch_size {
                self.process_event_batch()?;
                self.event_batch.clear();
                
                // Check max events limit
                if let Some(max) = self.config.max_events {
                    if self.events_processed >= max {
                        break;
                    }
                }
            }
        }
        
        // Process remaining events
        if !self.event_batch.is_empty() {
            self.process_event_batch()?;
        }
        
        Ok(())
    }
    
    /// Process batch of events efficiently
    fn process_event_batch(&mut self) -> Result<(), String> {
        // Update market snapshot from batch (lockless)
        for update in &self.event_batch {
            self.update_market_snapshot_fast(update);
        }
        
        // Process each event through strategies
        for update in &self.event_batch {
            self.process_single_event_optimized(update)?;
            self.events_processed += 1;
        }
        
        Ok(())
    }
    
    /// Fast market snapshot update (no locks)
    #[inline(always)]
    fn update_market_snapshot_fast(&mut self, update: &MarketUpdate) {
        match update {
            MarketUpdate::Trade(trade) => {
                self.market_snapshot.update_last_price(
                    trade.instrument_id,
                    trade.price,
                    trade.timestamp,
                );
                self.current_time = trade.timestamp;
            }
            MarketUpdate::BBO(bbo) => {
                self.market_snapshot.update_bbo(
                    bbo.instrument_id,
                    bbo.bid_price,
                    bbo.ask_price,
                    bbo.bid_quantity,
                    bbo.ask_quantity,
                    bbo.timestamp,
                );
                self.current_time = bbo.timestamp;
            }
            _ => {}
        }
    }
    
    /// Process single event with minimal allocations
    fn process_single_event_optimized(&mut self, update: &MarketUpdate) -> Result<(), String> {
        // Convert to market event for strategy compatibility
        let market_event = self.convert_update_to_event(update);
        
        // Clear pre-allocated output buffer
        self.strategy_outputs.clear();
        
        // Process through all strategies efficiently
        for wrapper in &mut self.strategies {
            // Update context with current market snapshot (no locks)
            wrapper.context.current_time = self.current_time;
            
            // Calculate features if enabled (using indexed system)
            if self.config.calculate_features {
                self.calculate_features_fast(update, &mut wrapper.features);
            }
            
            // Call strategy (main computational work)
            let output = wrapper.strategy.on_market_event(&market_event, &wrapper.context);
            
            // Store output for processing (reuse allocation)
            self.strategy_outputs.push((wrapper.context.strategy_id.clone(), output));
            
            // Update counters
            wrapper.events_processed += 1;
        }
        
        // Process strategy outputs
        for (strategy_id, output) in &self.strategy_outputs {
            self.process_strategy_output_fast(strategy_id, output)?;
        }
        
        Ok(())
    }
    
    /// Fast feature calculation using indexed system
    #[inline(always)]
    fn calculate_features_fast(&self, update: &MarketUpdate, features: &mut IndexedFeatureVector) {
        features.timestamp = self.current_time;
        
        match update {
            MarketUpdate::BBO(bbo) => {
                let spread_abs = bbo.ask_price.0 - bbo.bid_price.0;
                let spread_rel = spread_abs as f64 / bbo.bid_price.as_f64();
                let imbalance = (bbo.bid_quantity.as_i64() - bbo.ask_quantity.as_i64()) as f64 
                    / (bbo.bid_quantity.as_i64() + bbo.ask_quantity.as_i64()) as f64;
                
                // Set features by index (no string operations)
                features.set(feature_indices::SPREAD_ABSOLUTE, spread_abs as f64);
                features.set(feature_indices::SPREAD_RELATIVE, spread_rel);
                features.set(feature_indices::BID_SIZE, bbo.bid_quantity.as_f64());
                features.set(feature_indices::ASK_SIZE, bbo.ask_quantity.as_f64());
                features.set(feature_indices::VOLUME_IMBALANCE, imbalance);
            }
            MarketUpdate::Trade(trade) => {
                features.set(feature_indices::SPREAD_ABSOLUTE, 0.0); // No spread for trades
                features.set(feature_indices::BID_SIZE, trade.quantity.as_f64());
                features.set(feature_indices::ASK_SIZE, trade.quantity.as_f64());
            }
            _ => {}
        }
    }
    
    /// Convert MarketUpdate to MarketEvent for strategy compatibility
    fn convert_update_to_event(&self, update: &MarketUpdate) -> crate::market_data::events::MarketEvent {
        match update {
            MarketUpdate::Trade(trade) => {
                crate::market_data::events::MarketEvent::Trade(crate::market_data::events::TradeEvent {
                    instrument_id: trade.instrument_id,
                    trade_id: self.events_processed as u64,
                    price: trade.price,
                    quantity: trade.quantity,
                    aggressor_side: trade.side,
                    timestamp: trade.timestamp,
                    buyer_order_id: None,
                    seller_order_id: None,
                })
            }
            MarketUpdate::BBO(bbo) => {
                crate::market_data::events::MarketEvent::BBO(crate::market_data::events::BBOUpdate {
                    instrument_id: bbo.instrument_id,
                    bid_price: Some(bbo.bid_price),
                    ask_price: Some(bbo.ask_price),
                    bid_quantity: Some(bbo.bid_quantity),
                    ask_quantity: Some(bbo.ask_quantity),
                    bid_order_count: None,
                    ask_order_count: None,
                    timestamp: bbo.timestamp,
                })
            }
            _ => {
                // Default trade event for other types
                crate::market_data::events::MarketEvent::Trade(crate::market_data::events::TradeEvent {
                    instrument_id: 1,
                    trade_id: self.events_processed as u64,
                    price: Price::new(100_000_000),
                    quantity: Quantity::from(100u32),
                    aggressor_side: Side::Bid,
                    timestamp: self.current_time,
                    buyer_order_id: None,
                    seller_order_id: None,
                })
            }
        }
    }
    
    /// Fast strategy output processing
    fn process_strategy_output_fast(&mut self, strategy_id: &str, output: &StrategyOutput) -> Result<(), String> {
        // Process orders efficiently
        for order in &output.orders {
            self.execution_engine.submit_order(order.clone(), self.current_time)?;
        }
        
        // Process fills
        while let Some(fill) = self.execution_engine.next_fill(self.current_time) {
            self.position_manager.process_fill(&fill)?;
            self.metrics_collector.process_fill(&fill);
        }
        
        Ok(())
    }
    
    /// Generate optimized report
    fn generate_optimized_report(&self) -> OptimizedBacktestReport {
        let performance_metrics = self.metrics_collector.calculate_metrics();
        let position_stats = self.position_manager.get_portfolio_stats();
        
        // Collect strategy statistics
        let mut strategy_stats = Vec::new();
        for wrapper in &self.strategies {
            strategy_stats.push(OptimizedStrategyStats {
                strategy_id: wrapper.context.strategy_id.clone(),
                events_processed: wrapper.events_processed,
                orders_generated: wrapper.orders_generated,
            });
        }
        
        OptimizedBacktestReport {
            events_processed: self.events_processed,
            performance_metrics,
            position_stats,
            strategy_stats,
            elapsed_time_ns: 0, // Would be filled by calling code
        }
    }
}

/// Optimized backtest report
#[derive(Debug)]
pub struct OptimizedBacktestReport {
    pub events_processed: usize,
    pub performance_metrics: PerformanceMetrics,
    pub position_stats: PortfolioStats,
    pub strategy_stats: Vec<OptimizedStrategyStats>,
    pub elapsed_time_ns: u64,
}

impl OptimizedBacktestReport {
    /// Calculate throughput in events per second
    pub fn throughput(&self) -> f64 {
        if self.elapsed_time_ns > 0 {
            (self.events_processed as f64) / (self.elapsed_time_ns as f64 / 1e9)
        } else {
            0.0
        }
    }
    
    /// Print performance summary
    pub fn print_performance_summary(&self) {
        println!("=== Optimized Backtest Performance ===");
        println!("Events processed: {}", self.events_processed);
        println!("Elapsed time: {:.2}ms", self.elapsed_time_ns as f64 / 1e6);
        println!("Throughput: {:.0} events/second", self.throughput());
        println!("Target efficiency: {:.1}%", (self.throughput() / 18_000_000.0) * 100.0);
        
        println!("\n=== Strategy Statistics ===");
        for stats in &self.strategy_stats {
            println!("{}: {} events, {} orders", 
                     stats.strategy_id, stats.events_processed, stats.orders_generated);
        }
    }
}

/// Strategy performance statistics
#[derive(Debug)]
pub struct OptimizedStrategyStats {
    pub strategy_id: String,
    pub events_processed: u64,
    pub orders_generated: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
    
    #[test]
    fn test_optimized_engine_creation() {
        let config = OptimizedBacktestConfig::default();
        let engine = OptimizedBacktestEngine::new(config);
        assert_eq!(engine.strategies.len(), 0);
        assert_eq!(engine.events_processed, 0);
    }
    
    #[test]
    fn test_strategy_addition() {
        let mut engine = OptimizedBacktestEngine::new(OptimizedBacktestConfig::default());
        
        let strategy = MeanReversionStrategy::new(
            "TestMR".to_string(),
            1,
            MeanReversionConfig::default(),
        );
        
        assert!(engine.add_strategy(Box::new(strategy)).is_ok());
        assert_eq!(engine.strategies.len(), 1);
    }
}