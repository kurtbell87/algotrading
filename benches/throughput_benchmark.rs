//! Throughput benchmark for backtesting system
//!
//! Measures pure event processing speed to compare against 18M msg/s target

use algotrading::core::Side;
use algotrading::core::types::{InstrumentId, Price, Quantity};
use algotrading::market_data::events::{BBOUpdate, MarketEvent, TradeEvent};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use algotrading::strategy::output::OrderRequest;
use algotrading::strategy::{
    OrderSide, Strategy, StrategyConfig, StrategyContext, StrategyError, StrategyOutput,
};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::{Duration, Instant};

/// Simple strategy for benchmarking that does minimal work
struct BenchmarkStrategy {
    config: StrategyConfig,
    event_count: usize,
}

impl BenchmarkStrategy {
    fn new() -> Self {
        let config =
            StrategyConfig::new("Benchmark".to_string(), "Benchmark Strategy").with_instrument(1);

        Self {
            config,
            event_count: 0,
        }
    }
}

impl Strategy for BenchmarkStrategy {
    fn initialize(&mut self, _context: &StrategyContext) -> Result<(), StrategyError> {
        Ok(())
    }

    fn on_market_event(
        &mut self,
        _event: &MarketEvent,
        _context: &StrategyContext,
    ) -> StrategyOutput {
        self.event_count += 1;
        StrategyOutput::default()
    }

    fn config(&self) -> &StrategyConfig {
        &self.config
    }
}

/// Generate test market events
fn generate_market_events(num_events: usize) -> Vec<MarketEvent> {
    let mut events = Vec::with_capacity(num_events);
    let mut timestamp = 1_000_000;
    let mut price = 100_000_000; // 100.00 in fixed point

    for i in 0..num_events {
        timestamp += 100; // 100 microseconds between events
        price += ((i % 20) as i64) - 10;

        if i % 2 == 0 {
            // Trade event
            events.push(MarketEvent::Trade(TradeEvent {
                instrument_id: 1,
                trade_id: i as u64,
                price: Price::new(price),
                quantity: Quantity::from((100 + (i % 100)) as u32),
                aggressor_side: if i % 3 == 0 { Side::Bid } else { Side::Ask },
                timestamp,
                buyer_order_id: None,
                seller_order_id: None,
            }));
        } else {
            // BBO event
            let spread = 100;
            events.push(MarketEvent::BBO(BBOUpdate {
                instrument_id: 1,
                bid_price: Some(Price::new(price - spread / 2)),
                ask_price: Some(Price::new(price + spread / 2)),
                bid_quantity: Some(Quantity::from(200u32)),
                ask_quantity: Some(Quantity::from(200u32)),
                bid_order_count: None,
                ask_order_count: None,
                timestamp,
            }));
        }
    }

    events
}

/// Benchmark pure event processing (no strategies)
fn benchmark_pure_event_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("pure_event_processing");
    group.measurement_time(Duration::from_secs(5));

    for num_events in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_events),
            num_events,
            |b, &num_events| {
                let events = generate_market_events(num_events);

                b.iter(|| {
                    let start = Instant::now();

                    // Simulate pure event processing
                    for event in &events {
                        black_box(event);
                        // Minimal processing
                        let _timestamp = event.timestamp();
                        let _is_trade = event.is_trade();
                    }

                    let elapsed = start.elapsed();
                    let throughput = events.len() as f64 / elapsed.as_secs_f64();

                    // Store throughput for comparison
                    black_box(throughput);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark strategy event processing
fn benchmark_strategy_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_processing");
    group.measurement_time(Duration::from_secs(5));

    for num_events in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_events),
            num_events,
            |b, &num_events| {
                let events = generate_market_events(num_events);

                b.iter(|| {
                    let mut strategy = BenchmarkStrategy::new();

                    // Mock context
                    let context = StrategyContext {
                        strategy_id: "benchmark".to_string(),
                        current_time: 1_000_000,
                        position: algotrading::features::FeaturePosition {
                            quantity: 0,
                            avg_price: Price::new(100_000_000),
                            unrealized_pnl: 0.0,
                            realized_pnl: 0.0,
                        },
                        market_state: std::sync::Arc::new(std::sync::RwLock::new(
                            algotrading::backtest::market_state::MarketStateManager::new(),
                        )),
                        pending_orders: Vec::new(),
                        recent_trades: std::collections::VecDeque::new(),
                        session_stats: algotrading::strategy::SessionStats::default(),
                        risk_limits: algotrading::features::RiskLimits::default(),
                    };

                    let start = Instant::now();

                    for event in &events {
                        let _output = strategy.on_market_event(event, &context);
                    }

                    let elapsed = start.elapsed();
                    let throughput = events.len() as f64 / elapsed.as_secs_f64();

                    black_box(throughput);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark realistic strategy (mean reversion)
fn benchmark_realistic_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_strategy");
    group.measurement_time(Duration::from_secs(5));

    for num_events in [1_000, 10_000, 50_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_events),
            num_events,
            |b, &num_events| {
                let events = generate_market_events(num_events);

                b.iter(|| {
                    let mut strategy = MeanReversionStrategy::new(
                        "BenchmarkMR".to_string(),
                        1,
                        MeanReversionConfig {
                            lookback_period: 20,
                            entry_threshold: 1.5,
                            exit_threshold: 0.5,
                            max_position_size: 10,
                            order_size: 1,
                            use_limit_orders: false,
                            limit_order_offset_ticks: 1,
                        },
                    );

                    // Mock context
                    let context = StrategyContext {
                        strategy_id: "benchmark_mr".to_string(),
                        current_time: 1_000_000,
                        position: algotrading::features::FeaturePosition {
                            quantity: 0,
                            avg_price: Price::new(100_000_000),
                            unrealized_pnl: 0.0,
                            realized_pnl: 0.0,
                        },
                        market_state: std::sync::Arc::new(std::sync::RwLock::new(
                            algotrading::backtest::market_state::MarketStateManager::new(),
                        )),
                        pending_orders: Vec::new(),
                        recent_trades: std::collections::VecDeque::new(),
                        session_stats: algotrading::strategy::SessionStats::default(),
                        risk_limits: algotrading::features::RiskLimits::default(),
                    };

                    let start = Instant::now();

                    for event in &events {
                        let _output = strategy.on_market_event(event, &context);
                    }

                    let elapsed = start.elapsed();
                    let throughput = events.len() as f64 / elapsed.as_secs_f64();

                    black_box(throughput);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multiple strategies concurrently
fn benchmark_multiple_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_strategies");
    group.measurement_time(Duration::from_secs(8));

    let num_events = 10_000;
    let events = generate_market_events(num_events);

    for num_strategies in [1, 2, 3, 5, 10].iter() {
        group.throughput(Throughput::Elements(num_events as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_strategies),
            num_strategies,
            |b, &num_strategies| {
                b.iter(|| {
                    // Create multiple strategies
                    let mut strategies: Vec<Box<dyn Strategy>> = Vec::new();
                    for i in 0..num_strategies {
                        strategies.push(Box::new(MeanReversionStrategy::new(
                            format!("BenchmarkMR_{}", i),
                            1,
                            MeanReversionConfig {
                                lookback_period: 20 + (i * 5),
                                entry_threshold: 1.5,
                                exit_threshold: 0.5,
                                max_position_size: 5,
                                order_size: 1,
                                use_limit_orders: false,
                                limit_order_offset_ticks: 1,
                            },
                        )));
                    }

                    // Mock context
                    let context = StrategyContext {
                        strategy_id: "benchmark_multi".to_string(),
                        current_time: 1_000_000,
                        position: algotrading::features::FeaturePosition {
                            quantity: 0,
                            avg_price: Price::new(100_000_000),
                            unrealized_pnl: 0.0,
                            realized_pnl: 0.0,
                        },
                        market_state: std::sync::Arc::new(std::sync::RwLock::new(
                            algotrading::backtest::market_state::MarketStateManager::new(),
                        )),
                        pending_orders: Vec::new(),
                        recent_trades: std::collections::VecDeque::new(),
                        session_stats: algotrading::strategy::SessionStats::default(),
                        risk_limits: algotrading::features::RiskLimits::default(),
                    };

                    let start = Instant::now();

                    // Process events through all strategies
                    for event in &events {
                        for strategy in &mut strategies {
                            let _output = strategy.on_market_event(event, &context);
                        }
                    }

                    let elapsed = start.elapsed();
                    let total_events = events.len() * num_strategies;
                    let throughput = total_events as f64 / elapsed.as_secs_f64();

                    black_box(throughput);
                });
            },
        );
    }

    group.finish();
}

/// Print throughput results for comparison
fn print_throughput_summary() {
    println!("\n=== Throughput Comparison vs 18M msg/s Target ===");
    println!("Run the benchmarks to see detailed results:");
    println!("cargo bench --bench throughput_benchmark");
    println!("\nTarget: 18,000,000 messages/second");
    println!("Expected backtesting overhead:");
    println!("- Pure LOB replay: ~50-80% of target (9-14M msg/s)");
    println!("- Single strategy: ~30-50% of target (5-9M msg/s)");
    println!("- Multiple strategies: scales with strategy count");
    println!("- With features: ~20-40% of target (3-7M msg/s)");
}

criterion_group!(
    benches,
    benchmark_pure_event_processing,
    benchmark_strategy_processing,
    benchmark_realistic_strategy,
    benchmark_multiple_strategies
);
criterion_main!(benches);
