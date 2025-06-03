# Implementation Plan for Backtesting System

## Phase 1: Core Infrastructure (Week 1-2)

### 1.1 Strategy Trait and Basic Types
```rust
// src/strategy/mod.rs
pub mod traits;
pub mod context;
pub mod output;

// src/strategy/traits.rs
pub trait Strategy: Send + Sync {
    fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput;
    fn on_timer(&mut self, timestamp: Timestamp, context: &StrategyContext) -> StrategyOutput;
    fn config(&self) -> &StrategyConfig;
}

// src/strategy/context.rs
pub struct StrategyContext {
    pub strategy_id: StrategyId,
    pub current_time: Timestamp,
    pub position: Position,
    pub pending_orders: Vec<Order>,
    pub market_state: MarketStateView,
}

// src/strategy/output.rs
pub struct StrategyOutput {
    pub orders: Vec<OrderRequest>,
    pub cancellations: Vec<OrderId>,
    pub updates: Vec<OrderUpdate>,
    pub metrics: Option<StrategyMetrics>,
}
```

### 1.2 Market State Management
```rust
// src/backtest/market_state.rs
pub struct MarketState {
    order_books: HashMap<InstrumentId, OrderBook>,
    recent_trades: HashMap<InstrumentId, VecDeque<Trade>>,
    session_stats: HashMap<InstrumentId, SessionStatistics>,
}

// src/backtest/market_view.rs
pub struct MarketStateView {
    instrument_id: InstrumentId,
    state: Arc<RwLock<MarketState>>,
}
```

### 1.3 Basic Event Types
```rust
// src/backtest/events.rs
pub enum BacktestEvent {
    Market(MarketEvent),
    Timer(TimerEvent),
    OrderUpdate(OrderUpdateEvent),
    Fill(FillEvent),
}
```

## Phase 2: Execution Engine (Week 2-3)

### 2.1 Order Management
```rust
// src/execution/mod.rs
pub mod engine;
pub mod models;
pub mod order_book_sim;

// src/execution/engine.rs
pub struct ExecutionEngine {
    pending_orders: BTreeMap<Timestamp, Vec<PendingOrder>>,
    active_orders: HashMap<OrderId, Order>,
    order_book_sim: OrderBookSimulator,
    latency_model: Box<dyn LatencyModel>,
    fill_model: Box<dyn FillModel>,
}

// src/execution/models.rs
pub trait LatencyModel: Send + Sync {
    fn calculate_arrival_time(&self, submit_time: Timestamp, order_type: &OrderType) -> Timestamp;
}

pub trait FillModel: Send + Sync {
    fn simulate_fill(&self, order: &Order, book_state: &OrderBook) -> Option<Fill>;
}
```

### 2.2 Realistic Fill Simulation
```rust
// src/execution/fill_models.rs
pub struct ProbabilisticFillModel {
    fill_probability_at_touch: f64,
    queue_model: QueuePositionModel,
}

pub enum QueuePositionModel {
    FIFO,           // First in, first out
    ProRata,        // Proportional to size
    SizeWeighted,   // Larger orders get priority
}
```

## Phase 3: Backtesting Engine (Week 3-4)

### 3.1 Main Backtesting Loop
```rust
// src/backtest/engine.rs
pub struct BacktestEngine {
    market_replay: MarketReplayEngine,
    strategy_router: StrategyRouter,
    execution_engine: ExecutionEngine,
    position_manager: PositionManager,
    metrics_collector: MetricsCollector,
}

impl BacktestEngine {
    pub async fn run(&mut self) -> Result<BacktestReport> {
        let (event_tx, event_rx) = crossbeam::channel::bounded(10000);
        
        // Spawn market replay thread
        let replay_handle = std::thread::spawn(move || {
            self.market_replay.replay_to_channel(event_tx)
        });
        
        // Main processing loop
        while let Ok(events) = event_rx.recv_timeout(Duration::milliseconds(100)) {
            self.process_event_batch(events)?;
        }
        
        // Generate final report
        Ok(self.metrics_collector.generate_report())
    }
}
```

### 3.2 Strategy Router
```rust
// src/backtest/router.rs
pub struct StrategyRouter {
    strategies: HashMap<StrategyId, Box<dyn Strategy>>,
    contexts: HashMap<StrategyId, StrategyContext>,
    subscriptions: HashMap<InstrumentId, Vec<StrategyId>>,
}

impl StrategyRouter {
    pub fn route_events(&mut self, events: &[MarketEvent]) -> Vec<(StrategyId, StrategyOutput)> {
        // Efficient event routing based on subscriptions
    }
}
```

## Phase 4: Performance Optimizations (Week 4-5)

### 4.1 Memory Pool Allocators
```rust
// src/utils/memory.rs
pub struct ObjectPool<T> {
    pool: Vec<T>,
    reset_fn: Box<dyn Fn(&mut T)>,
}

impl<T> ObjectPool<T> {
    pub fn acquire(&mut self) -> PooledObject<T> {
        // Get object from pool or allocate new
    }
}
```

### 4.2 Lock-Free Data Structures
```rust
// src/utils/concurrent.rs
pub struct LockFreeRingBuffer<T> {
    buffer: Vec<AtomicPtr<T>>,
    head: AtomicUsize,
    tail: AtomicUsize,
}
```

### 4.3 Batch Processing Optimizations
```rust
// src/backtest/batch_processor.rs
pub struct BatchProcessor {
    event_buffer: Vec<MarketEvent>,
    output_buffer: Vec<StrategyOutput>,
    batch_size: usize,
}
```

## Phase 5: Python Integration (Week 5-6)

### 5.1 Python Bridge Design
```rust
// src/python/bridge.rs
pub struct PythonBridge {
    feature_sender: Sender<FeatureBatch>,
    signal_receiver: Receiver<SignalBatch>,
    py_process: Option<Child>,
}

// src/python/protocol.rs
#[derive(Serialize, Deserialize)]
pub struct FeatureBatch {
    pub timestamp: Timestamp,
    pub features: Vec<f64>,
    pub metadata: HashMap<String, Value>,
}
```

### 5.2 Python Strategy Interface
```python
# python/strategy_base.py
from abc import ABC, abstractmethod
import numpy as np

class StrategyBase(ABC):
    @abstractmethod
    def on_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Process features and return trading signals"""
        pass
    
    @abstractmethod
    def on_market_close(self) -> None:
        """Called at end of trading session"""
        pass
```

### 5.3 Communication Protocol
```python
# python/bridge_server.py
import asyncio
import msgpack
from typing import AsyncIterator

class BridgeServer:
    def __init__(self, strategy: StrategyBase):
        self.strategy = strategy
        
    async def process_messages(self, reader: asyncio.StreamReader, 
                             writer: asyncio.StreamWriter):
        unpacker = msgpack.Unpacker()
        
        while True:
            data = await reader.read(4096)
            if not data:
                break
                
            unpacker.feed(data)
            for msg in unpacker:
                response = self.strategy.on_features(msg['features'])
                await self.send_response(writer, response)
```

## Phase 6: Analytics and Reporting (Week 6-7)

### 6.1 Metrics Collection
```rust
// src/analytics/metrics.rs
pub struct MetricsCollector {
    pnl_series: TimeSeries<f64>,
    position_series: TimeSeries<i64>,
    trade_log: Vec<TradeRecord>,
    order_stats: OrderStatistics,
}

// src/analytics/report.rs
pub struct BacktestReport {
    pub summary: PerformanceSummary,
    pub risk_metrics: RiskMetrics,
    pub execution_stats: ExecutionStatistics,
    pub detailed_trades: Option<Vec<TradeRecord>>,
}
```

### 6.2 Report Generation
```rust
// src/analytics/generator.rs
impl BacktestReport {
    pub fn to_html(&self) -> String {
        // Generate HTML report with charts
    }
    
    pub fn to_parquet(&self, path: &Path) -> Result<()> {
        // Save detailed data to parquet files
    }
}
```

## Phase 7: Testing and Validation (Week 7-8)

### 7.1 Unit Test Suite
```rust
// tests/unit/execution_tests.rs
#[test]
fn test_latency_model() {
    let model = FixedLatencyModel::new(Duration::microseconds(50));
    let arrival = model.calculate_arrival_time(
        Timestamp::from_nanos(1000),
        &OrderType::Market
    );
    assert_eq!(arrival, Timestamp::from_nanos(1050));
}
```

### 7.2 Integration Tests
```rust
// tests/integration/backtest_tests.rs
#[tokio::test]
async fn test_simple_strategy_backtest() {
    let engine = create_test_engine();
    let strategy = create_test_strategy();
    engine.add_strategy("test", strategy);
    
    let report = engine.run().await.unwrap();
    assert!(report.summary.total_pnl > 0.0);
}
```

### 7.3 Performance Benchmarks
```rust
// benches/throughput.rs
#[bench]
fn bench_event_processing(b: &mut Bencher) {
    let engine = create_engine();
    let events = generate_events(1_000_000);
    
    b.iter(|| {
        engine.process_events(&events)
    });
}
```

## Phase 8: Example Strategies (Week 8)

### 8.1 Simple Strategies
```rust
// src/strategies/examples/momentum.rs
pub struct MomentumStrategy {
    lookback: usize,
    threshold: f64,
    position_size: Quantity,
}

// src/strategies/examples/mean_reversion.rs
pub struct MeanReversionStrategy {
    window: usize,
    entry_z_score: f64,
    exit_z_score: f64,
}
```

### 8.2 ML Strategy Example
```rust
// src/strategies/examples/ml_strategy.rs
pub struct MLStrategy {
    feature_extractor: FeatureExtractor,
    python_bridge: PythonBridge,
    position_sizer: PositionSizer,
}
```

## Testing Milestones

### Milestone 1: Basic Functionality (Week 2)
- [ ] Strategy trait implementation
- [ ] Basic market event processing
- [ ] Simple order generation

### Milestone 2: Execution Simulation (Week 4)
- [ ] Order matching logic
- [ ] Latency simulation
- [ ] Fill models working

### Milestone 3: Full Backtest (Week 6)
- [ ] Complete backtest of simple strategy
- [ ] Performance metrics calculated
- [ ] Report generation

### Milestone 4: Python Integration (Week 7)
- [ ] Python bridge functional
- [ ] ML strategy example working
- [ ] Feature extraction pipeline

### Milestone 5: Performance Target (Week 8)
- [ ] 18M messages/second throughput
- [ ] < 10μs latency per event
- [ ] Memory usage stable

## Resource Requirements

### Development Environment
- Rust 1.75+
- Python 3.10+
- 32GB RAM minimum for testing
- SSD for market data storage

### Dependencies
```toml
# Core dependencies
tokio = { version = "1.35", features = ["full"] }
crossbeam = "0.8"
arrow = "50.0"
datafusion = "35.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
msgpack = "1.0"

# Python integration
pyo3 = { version = "0.20", features = ["extension-module"] }

# Testing
criterion = "0.5"
proptest = "1.4"
```

## Risk Mitigation

### Performance Risks
1. **Python bottleneck**: Mitigate with batching and async processing
2. **Memory pressure**: Use object pools and bounded channels
3. **Lock contention**: Design lock-free where possible

### Accuracy Risks
1. **Look-ahead bias**: Strict timestamp ordering
2. **Survivorship bias**: Include delisted contracts
3. **Execution assumptions**: Conservative fill models

## Success Criteria

1. **Performance**: 18M messages/second sustained
2. **Accuracy**: Results match production within 1%
3. **Usability**: New strategy implementation < 100 LOC
4. **Reliability**: 24-hour backtest without memory growth
5. **Extensibility**: Easy to add new features/models