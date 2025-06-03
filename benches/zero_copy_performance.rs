//! Zero-copy performance benchmark
//!
//! Tests the performance improvements from eliminating MboMsg cloning
//! and using optimized data structures

use algotrading::core::types::{InstrumentId, MarketUpdate, Price, Quantity, Side, Trade};
use algotrading::market_data::{FastFeatureVector, FastMarketState};
use algotrading::market_data::{
    LAST_PRICE_IDX, PRICE_RETURN_IDX, SPREAD_IDX, TRADE_COUNT_IDX, VOLUME_IDX,
};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::collections::HashMap;
use std::time::Instant;

/// Generate market events for benchmarking
fn generate_market_updates(num_events: usize) -> Vec<MarketUpdate> {
    let mut events = Vec::with_capacity(num_events);
    let mut timestamp = 1_000_000;
    let mut price = 100_000_000;

    for i in 0..num_events {
        timestamp += 100;
        price += ((i % 20) as i64) - 10;

        events.push(MarketUpdate::Trade(Trade {
            instrument_id: (i % 10) as InstrumentId + 1,
            price: Price::new(price),
            quantity: Quantity::from((100 + (i % 100)) as u32),
            side: if i % 2 == 0 { Side::Bid } else { Side::Ask },
            timestamp,
        }));
    }

    events
}

/// Benchmark 1: Compare fast vs standard HashMap performance
fn benchmark_hashmap_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashmap_performance");
    group.measurement_time(std::time::Duration::from_secs(3));

    let num_operations = 100_000;

    group.bench_function("standard_hashmap", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut map: HashMap<u32, f64> = HashMap::new();

            for i in 0..num_operations {
                map.insert(i % 1000, i as f64);
                black_box(map.get(&(i % 1000)));
            }

            let elapsed = start.elapsed();
            let throughput = num_operations as f64 / elapsed.as_secs_f64();

            println!("Standard HashMap: {:.0} ops/second", throughput);
            black_box(throughput);
        });
    });

    group.bench_function("hashbrown_hashmap", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut map: hashbrown::HashMap<u32, f64> = hashbrown::HashMap::new();

            for i in 0..num_operations {
                map.insert(i % 1000, i as f64);
                black_box(map.get(&(i % 1000)));
            }

            let elapsed = start.elapsed();
            let throughput = num_operations as f64 / elapsed.as_secs_f64();

            println!("Hashbrown HashMap: {:.0} ops/second", throughput);
            black_box(throughput);
        });
    });

    group.finish();
}

/// Benchmark 2: Fast market state vs traditional market state
fn benchmark_market_state_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("market_state");
    group.measurement_time(std::time::Duration::from_secs(3));

    let updates = generate_market_updates(10_000);

    group.bench_function("fast_market_state", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut state = FastMarketState::new();

            for update in &updates {
                state.update(update);
                black_box(state.get_last_price(1));
            }

            let elapsed = start.elapsed();
            let throughput = updates.len() as f64 / elapsed.as_secs_f64();

            println!("Fast market state: {:.0} updates/second", throughput);
            black_box(throughput);
        });
    });

    group.bench_function("traditional_market_access", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut prices: HashMap<InstrumentId, Price> = HashMap::new();
            let mut timestamps: HashMap<InstrumentId, u64> = HashMap::new();

            for update in &updates {
                match update {
                    MarketUpdate::Trade(trade) => {
                        prices.insert(trade.instrument_id, trade.price);
                        timestamps.insert(trade.instrument_id, trade.timestamp);
                        black_box(prices.get(&1));
                    }
                    _ => {}
                }
            }

            let elapsed = start.elapsed();
            let throughput = updates.len() as f64 / elapsed.as_secs_f64();

            println!("Traditional market state: {:.0} updates/second", throughput);
            black_box(throughput);
        });
    });

    group.finish();
}

/// Benchmark 3: Fast feature vector vs string-based features
fn benchmark_feature_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");
    group.measurement_time(std::time::Duration::from_secs(3));

    let updates = generate_market_updates(10_000);

    group.bench_function("fast_feature_vector", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut features = FastFeatureVector::new(0);

            for update in &updates {
                if let MarketUpdate::Trade(trade) = update {
                    features.update_from_trade(trade, None);
                    black_box(features.get(LAST_PRICE_IDX));
                    black_box(features.get(VOLUME_IDX));
                }
            }

            let elapsed = start.elapsed();
            let throughput = updates.len() as f64 / elapsed.as_secs_f64();

            println!("Fast feature vector: {:.0} updates/second", throughput);
            black_box(throughput);
        });
    });

    group.bench_function("string_based_features", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut features: HashMap<String, f64> = HashMap::new();

            for update in &updates {
                if let MarketUpdate::Trade(trade) = update {
                    features.insert("last_price".to_string(), trade.price.as_f64());
                    features.insert("volume".to_string(), trade.quantity.as_f64());
                    black_box(features.get("last_price"));
                    black_box(features.get("volume"));
                }
            }

            let elapsed = start.elapsed();
            let throughput = updates.len() as f64 / elapsed.as_secs_f64();

            println!("String-based features: {:.0} updates/second", throughput);
            black_box(throughput);
        });
    });

    group.finish();
}

/// Benchmark 4: Event cloning overhead
fn benchmark_event_cloning_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_cloning");
    group.measurement_time(std::time::Duration::from_secs(3));

    let events = generate_market_updates(50_000);

    group.bench_function("with_cloning", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut cloned_events = Vec::with_capacity(events.len());

            for event in &events {
                cloned_events.push(event.clone()); // Simulate current bottleneck
                black_box(&cloned_events.last());
            }

            let elapsed = start.elapsed();
            let throughput = events.len() as f64 / elapsed.as_secs_f64();

            println!("With cloning: {:.0} events/second", throughput);
            black_box(throughput);
        });
    });

    group.bench_function("without_cloning", |b| {
        b.iter(|| {
            let start = Instant::now();

            for event in &events {
                black_box(event); // Just reference, no cloning
            }

            let elapsed = start.elapsed();
            let throughput = events.len() as f64 / elapsed.as_secs_f64();

            println!("Without cloning: {:.0} events/second", throughput);
            black_box(throughput);
        });
    });

    group.finish();
}

/// Benchmark 5: Combined optimizations test
fn benchmark_combined_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_optimizations");
    group.measurement_time(std::time::Duration::from_secs(5));

    for num_events in [10_000, 100_000, 500_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_events),
            num_events,
            |b, &num_events| {
                let events = generate_market_updates(num_events);

                b.iter(|| {
                    let start = Instant::now();

                    // Simulate optimized processing pipeline
                    let mut market_state = FastMarketState::new();
                    let mut features = FastFeatureVector::new(0);

                    // Process events with minimal overhead
                    for event in &events {
                        // Update market state (O(1) hashbrown operations)
                        market_state.update(event);

                        // Update features (array indexing)
                        if let MarketUpdate::Trade(trade) = event {
                            features.update_from_trade(
                                trade,
                                market_state.get_last_price(trade.instrument_id),
                            );
                        }

                        // Simulate strategy decision (minimal work)
                        black_box(features.get(LAST_PRICE_IDX));
                        black_box(market_state.get_last_price(1));
                    }

                    let elapsed = start.elapsed();
                    let throughput = events.len() as f64 / elapsed.as_secs_f64();
                    let efficiency = (throughput / 18_000_000.0) * 100.0;

                    if num_events == 500_000 {
                        println!(
                            "\\nCombined optimizations: {:.0} events/second ({:.1}% of 18M target)",
                            throughput, efficiency
                        );
                    }

                    black_box(throughput);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 6: Target performance test
fn benchmark_target_performance(c: &mut Criterion) {
    println!("\\n=== Zero-Copy Performance Target Test ===");
    println!("Goal: Achieve >15M events/s (>83% of 18M target)");
    println!("Previous: 3.1M events/s (17% of target)");

    let mut group = c.benchmark_group("target_performance");
    group.measurement_time(std::time::Duration::from_secs(10));

    let events = generate_market_updates(1_000_000);

    group.bench_function("optimized_pipeline", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Pre-allocate everything
            let mut market_state = FastMarketState::new();
            let mut features = FastFeatureVector::new(0);
            let mut decision_count = 0u64;

            // Batch process for cache efficiency
            const BATCH_SIZE: usize = 1000;
            for chunk in events.chunks(BATCH_SIZE) {
                for event in chunk {
                    // Optimized market state update
                    market_state.update(event);

                    // Fast feature calculation
                    if let MarketUpdate::Trade(trade) = event {
                        let prev_price = market_state.get_last_price(trade.instrument_id);
                        features.update_from_trade(trade, prev_price);

                        // Simulate strategy decision (minimal overhead)
                        let price = features.get(LAST_PRICE_IDX);
                        let volume = features.get(VOLUME_IDX);
                        if price > 100.0 && volume > 500.0 {
                            decision_count += 1;
                        }
                    }
                }
            }

            let elapsed = start.elapsed();
            let throughput = events.len() as f64 / elapsed.as_secs_f64();
            let efficiency = (throughput / 18_000_000.0) * 100.0;

            println!(
                "\\nOptimized pipeline: {:.0} events/second ({:.1}% of 18M target)",
                throughput, efficiency
            );
            println!("Decisions made: {}", decision_count);

            // Check if we hit our target
            if throughput >= 15_000_000.0 {
                println!("🎯 SUCCESS: Achieved target performance (>15M events/s)!");
            } else if throughput >= 10_000_000.0 {
                println!(
                    "✅ GOOD: Significant improvement ({:.1}x vs 3.1M baseline)",
                    throughput / 3_100_000.0
                );
            } else {
                println!("❌ NEEDS WORK: Still below 10M events/s target");
            }

            black_box(throughput);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_hashmap_performance,
    benchmark_market_state_performance,
    benchmark_feature_performance,
    benchmark_event_cloning_overhead,
    benchmark_combined_optimizations,
    benchmark_target_performance
);
criterion_main!(benches);
