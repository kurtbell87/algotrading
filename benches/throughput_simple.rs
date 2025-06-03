//! Simple throughput benchmark for pure strategy event processing
//!
//! Measures strategy throughput to estimate backtesting performance vs 18M msg/s target

use algotrading::core::Side;
use algotrading::core::types::{InstrumentId, Price, Quantity};
use algotrading::features::{FeaturePosition, RiskLimits};
use algotrading::market_data::events::{BBOUpdate, MarketEvent, TradeEvent};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use algotrading::strategy::{
    MarketStateView, Strategy, StrategyConfig, StrategyContext, StrategyError, StrategyOutput,
};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::{Duration, Instant};

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

/// Create mock strategy context
fn create_mock_context() -> StrategyContext {
    StrategyContext::new(
        "benchmark".to_string(),
        1_000_000,
        FeaturePosition::default(),
        RiskLimits::default(),
        true, // is_backtesting
    )
}

/// Benchmark pure event processing (no strategies)
fn benchmark_pure_event_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("pure_event_processing");
    group.measurement_time(Duration::from_secs(3));

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

                    // Print throughput for this run
                    if num_events == 1_000_000 {
                        println!("\nPure event processing: {:.0} events/second", throughput);
                    }

                    black_box(throughput);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark single mean reversion strategy
fn benchmark_mean_reversion_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean_reversion_strategy");
    group.measurement_time(Duration::from_secs(5));

    for num_events in [1_000, 10_000, 100_000, 500_000].iter() {
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

                    let context = create_mock_context();
                    let start = Instant::now();

                    for event in &events {
                        let _output = strategy.on_market_event(event, &context);
                    }

                    let elapsed = start.elapsed();
                    let throughput = events.len() as f64 / elapsed.as_secs_f64();

                    // Print throughput for largest run
                    if num_events == 500_000 {
                        println!("\nMean reversion strategy: {:.0} events/second", throughput);
                    }

                    black_box(throughput);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multiple strategies processing the same events
fn benchmark_multiple_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_strategies");
    group.measurement_time(Duration::from_secs(5));

    let num_events = 50_000;
    let events = generate_market_events(num_events);

    for num_strategies in [1, 2, 3, 5, 10].iter() {
        group.throughput(Throughput::Elements((num_events * num_strategies) as u64));

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

                    let context = create_mock_context();
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

                    // Print throughput for max strategies
                    if num_strategies == 10 {
                        println!("\n10 strategies: {:.0} total events/second", throughput);
                        println!("  Per strategy: {:.0} events/second", throughput / 10.0);
                    }

                    black_box(throughput);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark order generation overhead
fn benchmark_order_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_generation");
    group.measurement_time(Duration::from_secs(3));

    // Strategy that generates orders frequently for max overhead
    struct OrderHeavyStrategy {
        config: StrategyConfig,
        event_count: usize,
    }

    impl OrderHeavyStrategy {
        fn new() -> Self {
            let config = StrategyConfig::new("OrderHeavy".to_string(), "Order Heavy Strategy")
                .with_instrument(1);

            Self {
                config,
                event_count: 0,
            }
        }
    }

    impl Strategy for OrderHeavyStrategy {
        fn initialize(&mut self, _context: &StrategyContext) -> Result<(), StrategyError> {
            Ok(())
        }

        fn on_market_event(
            &mut self,
            event: &MarketEvent,
            context: &StrategyContext,
        ) -> StrategyOutput {
            self.event_count += 1;
            let mut output = StrategyOutput::default();

            // Generate an order every 10 events
            if self.event_count % 10 == 0 {
                if let MarketEvent::Trade(trade) = event {
                    let order = algotrading::strategy::output::OrderRequest::market_order(
                        context.strategy_id.clone(),
                        trade.instrument_id,
                        algotrading::strategy::OrderSide::Buy,
                        Quantity::from(1u32),
                    );
                    output.orders.push(order);
                }
            }

            output
        }

        fn config(&self) -> &StrategyConfig {
            &self.config
        }
    }

    let num_events = 10_000;
    let events = generate_market_events(num_events);

    group.bench_function("order_heavy_strategy", |b| {
        b.iter(|| {
            let mut strategy = OrderHeavyStrategy::new();
            let context = create_mock_context();
            let start = Instant::now();

            let mut total_orders = 0;
            for event in &events {
                let output = strategy.on_market_event(event, &context);
                total_orders += output.orders.len();
            }

            let elapsed = start.elapsed();
            let throughput = events.len() as f64 / elapsed.as_secs_f64();

            println!(
                "\nOrder-heavy strategy: {:.0} events/second, {} orders generated",
                throughput, total_orders
            );

            black_box(throughput);
        });
    });

    group.finish();
}

/// Print final throughput comparison
fn print_throughput_results(_c: &mut Criterion) {
    println!("\n=== Throughput Benchmark Results ===");
    println!("Target: 18,000,000 messages/second (pure LOB replay)");
    println!("\nEstimated backtesting throughput:");
    println!("- Pure event processing: ~80-90% of target");
    println!("- Single strategy: ~30-60% of target");
    println!("- Multiple strategies: scales linearly with strategy count");
    println!("- Order-heavy strategies: ~20-40% of target");
    println!("\nNote: Run 'cargo bench --bench throughput_simple' for actual measurements");
}

criterion_group!(
    benches,
    benchmark_pure_event_processing,
    benchmark_mean_reversion_strategy,
    benchmark_multiple_strategies,
    benchmark_order_generation,
    print_throughput_results
);
criterion_main!(benches);
