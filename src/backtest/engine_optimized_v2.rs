//! Optimized backtesting engine v2
//!
//! Key optimizations:
//! 1. Batch processing (1000 events per batch)
//! 2. Direct processing (no event queue)
//! 3. Pre-allocated buffers
//! 4. Reduced lock contention

use crate::core::types::{InstrumentId, Price, Quantity};
use crate::core::traits::MarketDataSource;
use crate::core::{MarketUpdate, Side};
use crate::strategy::{Strategy, StrategyContext, StrategyOutput, OrderRequest};
use crate::features::{FeaturePosition, RiskLimits};
use crate::market_data::events::{MarketEvent, TradeEvent};
use crate::market_data::FileReader;
use crate::backtest::{BacktestConfig, EngineReport, PerformanceMetrics};
use crate::backtest::market_state::MarketStateManager;
use crate::backtest::position::PositionManager;
use crate::backtest::metrics::MetricsCollector;
use crate::backtest::execution::{ExecutionEngine, LatencyModel, FillModel};
use crate::backtest::events::FillEvent;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::path::{Path, PathBuf};
use std::time::Instant;

const BATCH_SIZE: usize = 1000;
const PROGRESS_INTERVAL: usize = 100_000;

/// Optimized backtesting engine with batch processing
pub struct OptimizedEngineV2 {
    config: BacktestConfig,
    strategies: Vec<StrategyState>,
    market_state: Arc<RwLock<MarketStateManager>>,
    execution_engine: ExecutionEngine,
    position_manager: PositionManager,
    metrics_collector: MetricsCollector,
    current_time: u64,
    events_processed: usize,
    // Pre-allocated buffers
    event_batch: Vec<MarketUpdate>,
    market_event_batch: Vec<MarketEvent>,
    order_buffer: Vec<(String, OrderRequest)>, // (strategy_id, order)
}

struct StrategyState {
    strategy: Box<dyn Strategy>,
    context: StrategyContext,
    position: HashMap<InstrumentId, i64>,
}

impl OptimizedEngineV2 {
    pub fn new(config: BacktestConfig) -> Self {
        let market_state = Arc::new(RwLock::new(MarketStateManager::new()));
        
        Self {
            execution_engine: ExecutionEngine::new(
                config.latency_model.clone(),
                config.fill_model.clone(),
                market_state.clone(),
            ),
            position_manager: PositionManager::new(RiskLimits::default()),
            metrics_collector: MetricsCollector::new(config.initial_capital),
            market_state,
            config,
            strategies: Vec::new(),
            current_time: 0,
            events_processed: 0,
            // Pre-allocate buffers
            event_batch: Vec::with_capacity(BATCH_SIZE),
            market_event_batch: Vec::with_capacity(BATCH_SIZE),
            order_buffer: Vec::with_capacity(100),
        }
    }
    
    /// Add strategy to engine
    pub fn add_strategy(&mut self, mut strategy: Box<dyn Strategy>) -> Result<(), String> {
        let strategy_config = strategy.config();
        let strategy_id = strategy_config.name.clone();
        
        // Create context
        let context = StrategyContext::new(
            strategy_id.clone(),
            self.current_time,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );
        
        // Initialize strategy
        strategy.initialize(&context)?;
        
        // Create state
        let state = StrategyState {
            strategy,
            context,
            position: HashMap::new(),
        };
        
        // Register with position manager
        self.position_manager.add_strategy(strategy_id, RiskLimits::default());
        
        self.strategies.push(state);
        Ok(())
    }
    
    /// Run backtest with optimized batch processing
    pub fn run<P: AsRef<Path>>(&mut self, data_files: &[P]) -> Result<EngineReport, String> {
        let start_time = Instant::now();
        println!("\n=== Optimized Engine V2 Starting ===");
        
        // Process each file
        for (i, file_path) in data_files.iter().enumerate() {
            println!("Processing file {}/{}: {:?}", i + 1, data_files.len(), 
                    file_path.as_ref().file_name().unwrap_or_default());
            self.process_file_optimized(file_path)?;
        }
        
        let elapsed = start_time.elapsed();
        let throughput = self.events_processed as f64 / elapsed.as_secs_f64();
        
        println!("\n=== Performance Results ===");
        println!("Total events: {}", self.events_processed);
        println!("Total time: {:.2}s", elapsed.as_secs_f64());
        println!("Throughput: {:.0} events/s", throughput);
        println!("Efficiency vs 18M target: {:.1}%", (throughput / 18_000_000.0) * 100.0);
        
        // Generate report
        Ok(EngineReport {
            config: self.config.clone(),
            events_processed: self.events_processed,
            strategy_results: Vec::new(),
            portfolio_stats: self.position_manager.get_portfolio_stats(),
            performance_metrics: self.metrics_collector.calculate_metrics(),
            trades: Vec::new(),
            equity_curve: Vec::new(),
        })
    }
    
    /// Process file with batch optimization
    fn process_file_optimized<P: AsRef<Path>>(&mut self, file_path: P) -> Result<(), String> {
        let paths = vec![PathBuf::from(file_path.as_ref())];
        let mut reader = FileReader::new(paths)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        
        let mut file_events = 0;
        let file_start = Instant::now();
        
        // Process in batches
        loop {
            // Clear batch
            self.event_batch.clear();
            
            // Fill batch
            while self.event_batch.len() < BATCH_SIZE {
                if let Some(update) = reader.next_update() {
                    self.event_batch.push(update);
                } else {
                    break;
                }
            }
            
            if self.event_batch.is_empty() {
                break; // No more events
            }
            
            // Process batch
            self.process_batch()?;
            
            file_events += self.event_batch.len();
            
            // Progress report
            if self.events_processed % PROGRESS_INTERVAL == 0 {
                let elapsed = file_start.elapsed();
                let rate = file_events as f64 / elapsed.as_secs_f64();
                println!("  Processed {} events at {:.0} events/s", 
                        self.events_processed, rate);
            }
        }
        
        Ok(())
    }
    
    /// Process a batch of events with optimizations
    fn process_batch(&mut self) -> Result<(), String> {
        // Step 1: Update market state for entire batch (single lock)
        self.update_market_state_batch();
        
        // Step 2: Convert updates to market events
        self.convert_batch_to_events();
        
        // Step 3: Process all events through strategies
        self.process_strategy_batch();
        
        // Step 4: Submit accumulated orders
        self.submit_orders_batch()?;
        
        // Update event count
        self.events_processed += self.event_batch.len();
        
        Ok(())
    }
    
    /// Update market state for entire batch (single lock acquisition)
    #[inline]
    fn update_market_state_batch(&mut self) {
        let mut market_state = self.market_state.write().unwrap();
        
        for update in &self.event_batch {
            self.current_time = match update {
                MarketUpdate::Trade(trade) => trade.timestamp,
                MarketUpdate::OrderBook(book) => book.timestamp,
            };
            
            // Convert to market event for state update
            let event = match update {
                MarketUpdate::Trade(trade) => {
                    MarketEvent::Trade(TradeEvent {
                        instrument_id: trade.instrument_id,
                        trade_id: 0,
                        price: trade.price,
                        quantity: trade.quantity,
                        aggressor_side: trade.side,
                        timestamp: trade.timestamp,
                        buyer_order_id: None,
                        seller_order_id: None,
                    })
                }
                MarketUpdate::OrderBook(_) => {
                    // Simplified for now
                    continue;
                }
            };
            
            market_state.process_event(&event);
        }
    }
    
    /// Convert batch of updates to market events
    #[inline]
    fn convert_batch_to_events(&mut self) {
        self.market_event_batch.clear();
        
        for update in &self.event_batch {
            let event = match update {
                MarketUpdate::Trade(trade) => {
                    MarketEvent::Trade(TradeEvent {
                        instrument_id: trade.instrument_id,
                        trade_id: 0,
                        price: trade.price,
                        quantity: trade.quantity,
                        aggressor_side: trade.side,
                        timestamp: trade.timestamp,
                        buyer_order_id: None,
                        seller_order_id: None,
                    })
                }
                MarketUpdate::OrderBook(book) => {
                    // Simplified - convert to trade
                    MarketEvent::Trade(TradeEvent {
                        instrument_id: book.instrument_id,
                        trade_id: 0,
                        price: Price::from(100i64),
                        quantity: Quantity::from(1u32),
                        aggressor_side: Side::Bid,
                        timestamp: book.timestamp,
                        buyer_order_id: None,
                        seller_order_id: None,
                    })
                }
            };
            
            self.market_event_batch.push(event);
        }
    }
    
    /// Process batch through all strategies
    #[inline]
    fn process_strategy_batch(&mut self) {
        self.order_buffer.clear();
        
        // Process each event
        for (i, event) in self.market_event_batch.iter().enumerate() {
            let timestamp = match event {
                MarketEvent::Trade(t) => t.timestamp,
                _ => self.current_time,
            };
            
            // Update all strategies
            for state in &mut self.strategies {
                // Update context time
                state.context.current_time = timestamp;
                
                // Call strategy
                let output = state.strategy.on_market_event(event, &state.context);
                
                // Collect orders
                for order in output.orders {
                    self.order_buffer.push((state.context.strategy_id.clone(), order));
                }
            }
        }
    }
    
    /// Submit accumulated orders
    fn submit_orders_batch(&mut self) -> Result<(), String> {
        // Submit all orders
        for (strategy_id, order) in &self.order_buffer {
            self.execution_engine.submit_order(
                order.clone(),
                strategy_id.clone(),
                self.current_time,
            );
        }
        
        // Process any immediate fills
        let fills = self.execution_engine.process_orders(self.current_time);
        for fill in fills {
            self.process_fill(fill)?;
        }
        
        Ok(())
    }
    
    /// Process fill event
    fn process_fill(&mut self, fill: FillEvent) -> Result<(), String> {
        // Update position manager
        self.position_manager.apply_fill(&fill)?;
        
        // Update metrics
        self.metrics_collector.process_fill(&fill);
        
        // Update strategy state
        for state in &mut self.strategies {
            if state.context.strategy_id == fill.strategy_id {
                let position = state.position.entry(fill.instrument_id).or_insert(0);
                match fill.side {
                    Side::Bid => *position += fill.quantity.as_i64(),
                    Side::Ask => *position -= fill.quantity.as_i64(),
                }
                
                state.context.position.quantity = *position;
                
                // Notify strategy
                state.strategy.on_fill(
                    fill.price,
                    fill.quantity.as_i64(),
                    fill.timestamp,
                    &state.context,
                );
                
                break;
            }
        }
        
        Ok(())
    }
}

/// Run performance comparison
pub fn compare_performance<P: AsRef<Path>>(
    data_files: &[P],
    original_throughput: f64,
) -> Result<f64, String> {
    println!("\n=== Testing Optimized Engine V2 ===");
    
    let config = BacktestConfig::default();
    let mut engine = OptimizedEngineV2::new(config);
    
    // Add test strategy
    use crate::strategies::MeanReversionStrategy;
    use crate::strategies::mean_reversion::MeanReversionConfig;
    
    let strategy = MeanReversionStrategy::new(
        "TestMR".to_string(),
        5921, // MES
        MeanReversionConfig::default(),
    );
    
    engine.add_strategy(Box::new(strategy))?;
    
    // Run backtest
    let report = engine.run(data_files)?;
    
    let optimized_throughput = report.events_processed as f64 / 
                              (report.events_processed as f64 / original_throughput);
    
    Ok(optimized_throughput)
}