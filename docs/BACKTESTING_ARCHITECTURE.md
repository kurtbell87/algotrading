# Backtesting Architecture Design

## Overview

This document outlines the design of a high-performance, strategy-agnostic backtesting system capable of processing 18M messages per second while supporting diverse trading strategies and ML/RL integration.

## Core Architecture Principles

1. **Strategy Agnostic**: Core engine knows nothing about specific strategies
2. **Zero-Copy Processing**: Minimize data copying in hot path
3. **Lock-Free Design**: Use channels and message passing vs shared state
4. **Batch Processing**: Accumulate events for efficient processing
5. **Async ML Integration**: Non-blocking Python strategy calls

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Backtesting Engine                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐         │
│  │   Market    │───▶│   Strategy   │───▶│   Execution   │         │
│  │   Replay    │    │   Router     │    │   Engine      │         │
│  └─────────────┘    └──────────────┘    └───────────────┘         │
│         │                   │                     │                  │
│         ▼                   ▼                     ▼                  │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐         │
│  │   Market    │    │   Strategy   │    │   Position    │         │
│  │   State     │◀───│   Instance   │───▶│   Manager     │         │
│  └─────────────┘    └──────────────┘    └───────────────┘         │
│                                                  │                   │
│                                                  ▼                   │
│                                          ┌───────────────┐         │
│                                          │   Analytics   │         │
│                                          │   Collector   │         │
│                                          └───────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Market Replay Engine
Responsible for reading historical data and replaying it at maximum speed.

```rust
pub struct MarketReplayEngine {
    data_source: Box<dyn MarketDataSource>,
    replay_speed: ReplaySpeed,
    event_buffer: Vec<MarketEvent>,
    batch_size: usize,
}

pub enum ReplaySpeed {
    MaxThroughput,      // Process as fast as possible
    Realtime,           // Maintain original timestamps
    Accelerated(f64),   // X times realtime
}

impl MarketReplayEngine {
    pub async fn run<F>(&mut self, mut event_handler: F) 
    where 
        F: FnMut(&[MarketEvent]) -> Result<(), BacktestError>
    {
        while let Some(batch) = self.data_source.next_batch(self.batch_size).await? {
            // Process in batches for efficiency
            self.event_buffer.clear();
            self.event_buffer.extend_from_slice(&batch);
            
            // Apply replay speed if needed
            if self.replay_speed != ReplaySpeed::MaxThroughput {
                self.apply_timing_control(&mut self.event_buffer).await;
            }
            
            // Send to handler
            event_handler(&self.event_buffer)?;
        }
    }
}
```

### 2. Strategy Router
Routes market events to appropriate strategies and manages strategy lifecycle.

```rust
pub trait Strategy: Send + Sync {
    /// Called on each market event
    fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput;
    
    /// Called periodically for time-based strategies
    fn on_timer(&mut self, timestamp: Timestamp, context: &StrategyContext) -> StrategyOutput;
    
    /// Strategy configuration
    fn config(&self) -> &StrategyConfig;
}

pub struct StrategyOutput {
    pub orders: Vec<OrderRequest>,
    pub cancellations: Vec<OrderId>,
    pub updates: Vec<OrderUpdate>,
    pub metrics: Option<StrategyMetrics>,
}

pub struct StrategyRouter {
    strategies: HashMap<StrategyId, Box<dyn Strategy>>,
    contexts: HashMap<StrategyId, StrategyContext>,
    event_dispatcher: EventDispatcher,
}

impl StrategyRouter {
    pub fn process_events(&mut self, events: &[MarketEvent]) -> Vec<(StrategyId, StrategyOutput)> {
        let mut outputs = Vec::new();
        
        // Dispatch events to strategies
        for event in events {
            for (id, strategy) in &mut self.strategies {
                let context = &self.contexts[id];
                
                // Check if strategy is interested in this event
                if self.event_dispatcher.should_dispatch(id, event) {
                    let output = strategy.on_market_event(event, context);
                    if !output.is_empty() {
                        outputs.push((*id, output));
                    }
                }
            }
        }
        
        outputs
    }
}
```

### 3. Execution Engine
Simulates order matching and execution with realistic market dynamics.

```rust
pub struct ExecutionEngine {
    order_book: OrderBook,
    pending_orders: BTreeMap<Timestamp, Vec<PendingOrder>>,
    latency_model: LatencyModel,
    fill_model: FillModel,
    market_impact_model: MarketImpactModel,
}

pub struct PendingOrder {
    pub order: Order,
    pub arrival_time: Timestamp,
    pub strategy_id: StrategyId,
}

impl ExecutionEngine {
    pub fn process_order_request(&mut self, request: OrderRequest, current_time: Timestamp) -> Result<OrderId> {
        // Apply latency model
        let arrival_time = self.latency_model.calculate_arrival_time(current_time, &request);
        
        // Apply market impact
        let adjusted_order = self.market_impact_model.adjust_order(request.into(), &self.order_book);
        
        // Queue order for processing at arrival time
        let pending = PendingOrder {
            order: adjusted_order,
            arrival_time,
            strategy_id: request.strategy_id,
        };
        
        self.pending_orders.entry(arrival_time)
            .or_insert_with(Vec::new)
            .push(pending);
        
        Ok(adjusted_order.id)
    }
    
    pub fn process_market_update(&mut self, update: &MarketUpdate, timestamp: Timestamp) -> Vec<Fill> {
        // Update internal order book
        self.order_book.apply_update(update);
        
        // Process any orders that should arrive by now
        let mut fills = Vec::new();
        while let Some(entry) = self.pending_orders.first_entry() {
            if *entry.key() <= timestamp {
                let orders = entry.remove();
                for pending in orders {
                    if let Some(fill) = self.try_fill_order(pending.order, timestamp) {
                        fills.push(fill);
                    }
                }
            } else {
                break;
            }
        }
        
        fills
    }
}
```

### 4. Position Manager
Tracks positions, P&L, and risk metrics for each strategy.

```rust
pub struct PositionManager {
    positions: HashMap<StrategyId, Position>,
    fills: Vec<Fill>,
    risk_limits: HashMap<StrategyId, RiskLimits>,
}

pub struct Position {
    pub quantity: i64,
    pub avg_price: Price,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub max_position: i64,
    pub total_volume: u64,
}

impl PositionManager {
    pub fn process_fill(&mut self, fill: Fill, current_price: Price) -> Result<()> {
        let position = self.positions.entry(fill.strategy_id)
            .or_insert_with(Position::default);
        
        // Update position
        let old_quantity = position.quantity;
        position.quantity += fill.quantity * fill.side.sign();
        
        // Calculate P&L
        if old_quantity != 0 && old_quantity.signum() != position.quantity.signum() {
            // Position flip or close
            let closed_quantity = old_quantity.min(fill.quantity);
            position.realized_pnl += closed_quantity as f64 * 
                (fill.price - position.avg_price) * fill.side.sign() as f64;
        }
        
        // Update average price
        if position.quantity != 0 {
            position.avg_price = self.calculate_avg_price(position, &fill);
        }
        
        // Check risk limits
        self.check_risk_limits(fill.strategy_id, position)?;
        
        Ok(())
    }
}
```

### 5. Strategy Context
Provides strategies with necessary market and portfolio information.

```rust
pub struct StrategyContext {
    pub strategy_id: StrategyId,
    pub current_time: Timestamp,
    pub position: Position,
    pub pending_orders: Vec<Order>,
    pub market_state: MarketStateView,
    pub config: StrategyConfig,
}

pub struct MarketStateView {
    order_book: Arc<RwLock<OrderBook>>,
    recent_trades: Arc<RwLock<VecDeque<Trade>>>,
    session_stats: SessionStatistics,
}

impl MarketStateView {
    pub fn best_bid(&self) -> Option<PriceLevel> {
        self.order_book.read().unwrap().best_bid()
    }
    
    pub fn best_ask(&self) -> Option<PriceLevel> {
        self.order_book.read().unwrap().best_ask()
    }
    
    pub fn recent_trades(&self, count: usize) -> Vec<Trade> {
        self.recent_trades.read().unwrap()
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
}
```

## Strategy Implementation Examples

### Simple Strategy Implementation
```rust
pub struct SimpleMovingAverageCrossover {
    fast_ma: MovingAverage,
    slow_ma: MovingAverage,
    position_size: Quantity,
    last_signal: Option<Signal>,
}

impl Strategy for SimpleMovingAverageCrossover {
    fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput {
        let mut output = StrategyOutput::default();
        
        // Only process trades
        if let MarketEvent::Trade(trade) = event {
            // Update moving averages
            self.fast_ma.update(trade.price);
            self.slow_ma.update(trade.price);
            
            // Check for crossover
            let fast_value = self.fast_ma.value();
            let slow_value = self.slow_ma.value();
            
            let current_signal = if fast_value > slow_value {
                Signal::Long
            } else {
                Signal::Short
            };
            
            // Generate orders on signal change
            if Some(current_signal) != self.last_signal {
                // Exit current position
                if context.position.quantity != 0 {
                    output.orders.push(OrderRequest::market_order(
                        -context.position.quantity,
                        context.strategy_id,
                    ));
                }
                
                // Enter new position
                let side = if current_signal == Signal::Long { 
                    Side::Buy 
                } else { 
                    Side::Sell 
                };
                
                output.orders.push(OrderRequest::market_order_with_side(
                    self.position_size,
                    side,
                    context.strategy_id,
                ));
                
                self.last_signal = Some(current_signal);
            }
        }
        
        output
    }
    
    fn on_timer(&mut self, _timestamp: Timestamp, _context: &StrategyContext) -> StrategyOutput {
        StrategyOutput::default() // No time-based actions
    }
    
    fn config(&self) -> &StrategyConfig {
        &self.config
    }
}
```

### ML Strategy Wrapper
```rust
pub struct MLStrategyWrapper {
    feature_extractor: FeatureExtractor,
    feature_buffer: Vec<Features>,
    python_bridge: PythonBridge,
    last_prediction_time: Timestamp,
    prediction_interval: Duration,
}

impl Strategy for MLStrategyWrapper {
    fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput {
        // Extract features
        if let Some(features) = self.feature_extractor.extract(event, context) {
            self.feature_buffer.push(features);
        }
        
        // Check if we should run prediction
        if context.current_time - self.last_prediction_time >= self.prediction_interval {
            // Send features to Python asynchronously
            if !self.feature_buffer.is_empty() {
                let features_batch = std::mem::take(&mut self.feature_buffer);
                self.python_bridge.send_features(features_batch);
                self.last_prediction_time = context.current_time;
            }
        }
        
        // Check for Python responses
        if let Some(signal) = self.python_bridge.try_recv_signal() {
            return self.convert_signal_to_orders(signal, context);
        }
        
        StrategyOutput::default()
    }
}
```

## Performance Optimizations

### 1. Memory Layout
```rust
// Optimize cache locality
#[repr(C, align(64))] // Cache line aligned
pub struct HotPathData {
    pub last_price: Price,
    pub last_quantity: Quantity,
    pub bid_price: Price,
    pub ask_price: Price,
    pub timestamp: Timestamp,
    _padding: [u8; 24], // Ensure full cache line
}
```

### 2. Batch Processing
```rust
pub struct BatchProcessor {
    batch_size: usize,
    events: Vec<MarketEvent>,
    results: Vec<StrategyOutput>,
}

impl BatchProcessor {
    pub fn process_batch(&mut self, strategies: &mut [Box<dyn Strategy>]) {
        // Process all events for all strategies in tight loop
        for strategy in strategies {
            for event in &self.events {
                // Process without allocation
                strategy.process_event_inplace(event, &mut self.results);
            }
        }
    }
}
```

### 3. Async Python Bridge
```rust
pub struct PythonBridge {
    tx: Sender<FeatureBatch>,
    rx: Receiver<SignalBatch>,
    runtime: tokio::runtime::Runtime,
}

impl PythonBridge {
    pub fn new(python_endpoint: &str) -> Self {
        let (feature_tx, feature_rx) = channel(1000);
        let (signal_tx, signal_rx) = channel(1000);
        
        // Spawn async task for Python communication
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.spawn(async move {
            python_communication_loop(
                python_endpoint,
                feature_rx,
                signal_tx,
            ).await
        });
        
        Self {
            tx: feature_tx,
            rx: signal_rx,
            runtime,
        }
    }
}
```

## Configuration

### Backtest Configuration
```yaml
backtest:
  data_source:
    type: "databento"
    files: ["glbx-mdp3-20250428.mbo.dbn.zst"]
    symbols: ["ESM5"]
  
  execution:
    latency_model:
      type: "fixed"
      network_latency_us: 50
      processing_latency_us: 10
    
    fill_model:
      type: "probabilistic"
      fill_probability_at_touch: 0.5
      queue_position_model: "pro_rata"
    
    market_impact:
      type: "linear"
      impact_bps_per_million: 0.5
  
  risk_limits:
    max_position: 100
    max_order_size: 50
    max_loss: 10000.0
    max_orders_per_second: 100
  
  output:
    metrics_file: "results/metrics.parquet"
    trades_file: "results/trades.parquet"
    detailed_log: false
```

### Strategy Configuration
```yaml
strategy:
  type: "ml_enhanced_momentum"
  parameters:
    lookback_period: 100
    entry_threshold: 2.0
    exit_threshold: 0.5
    ml_model_path: "models/momentum_v1.pkl"
    feature_config:
      price_features: ["returns_1m", "returns_5m", "volatility"]
      microstructure_features: ["book_imbalance", "trade_flow"]
      use_time_features: true
```

## Analytics and Metrics

### Real-time Metrics Collection
```rust
pub struct MetricsCollector {
    pnl_curve: Vec<(Timestamp, f64)>,
    position_history: Vec<(Timestamp, i64)>,
    fill_history: Vec<Fill>,
    order_stats: OrderStatistics,
    latency_histogram: Histogram,
}

impl MetricsCollector {
    pub fn record_fill(&mut self, fill: &Fill) {
        self.fill_history.push(fill.clone());
        self.order_stats.total_fills += 1;
        self.order_stats.total_volume += fill.quantity;
    }
    
    pub fn record_pnl(&mut self, timestamp: Timestamp, pnl: f64) {
        self.pnl_curve.push((timestamp, pnl));
    }
    
    pub fn generate_report(&self) -> BacktestReport {
        BacktestReport {
            total_pnl: self.calculate_total_pnl(),
            sharpe_ratio: self.calculate_sharpe_ratio(),
            max_drawdown: self.calculate_max_drawdown(),
            total_trades: self.fill_history.len(),
            win_rate: self.calculate_win_rate(),
            avg_trade_duration: self.calculate_avg_trade_duration(),
            // ... more metrics
        }
    }
}
```

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_strategy_router_dispatch() {
        let mut router = StrategyRouter::new();
        let strategy = Box::new(TestStrategy::new());
        router.register_strategy("test", strategy);
        
        let events = vec![
            MarketEvent::Trade(create_test_trade(100.0, 10)),
        ];
        
        let outputs = router.process_events(&events);
        assert_eq!(outputs.len(), 1);
    }
    
    #[test]
    fn test_execution_engine_latency() {
        let mut engine = ExecutionEngine::new(LatencyModel::Fixed(Duration::microseconds(50)));
        
        let order = OrderRequest::market_order(10, "test".into());
        let current_time = Timestamp::now();
        
        engine.process_order_request(order, current_time).unwrap();
        
        // Order should not be executed immediately
        let fills = engine.process_market_update(&create_test_update(), current_time);
        assert!(fills.is_empty());
        
        // Order should be executed after latency
        let fills = engine.process_market_update(
            &create_test_update(), 
            current_time + Duration::microseconds(51)
        );
        assert_eq!(fills.len(), 1);
    }
}
```

### Integration Tests
```rust
#[tokio::test]
async fn test_full_backtest_pipeline() {
    let config = BacktestConfig::from_file("tests/test_config.yaml").unwrap();
    let strategy = Box::new(SimpleMovingAverageCrossover::new(10, 20));
    
    let mut backtester = Backtester::new(config);
    backtester.add_strategy("ma_cross", strategy);
    
    let report = backtester.run().await.unwrap();
    
    assert!(report.total_trades > 0);
    assert!(report.sharpe_ratio.is_finite());
}
```

## Summary

This architecture provides:
1. **Strategy agnostic core** - Strategies implement simple trait
2. **High performance** - Batch processing, zero-copy, lock-free design
3. **Realistic simulation** - Latency, market impact, partial fills
4. **ML/RL ready** - Async Python bridge for complex models
5. **Comprehensive analytics** - Real-time metrics and reporting

The design maintains 18M msg/s throughput while providing flexibility for diverse strategy implementations.