//! Focused performance benchmark to identify and fix the main bottlenecks
//!
//! This benchmark specifically targets the 5.8x performance gap

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use algotrading::core::types::{InstrumentId, Price, Quantity};
use algotrading::core::Side;
use algotrading::market_data::events::{MarketEvent, TradeEvent, BBOUpdate};
use algotrading::strategy::{Strategy, StrategyConfig, StrategyContext, StrategyOutput, StrategyError};
use algotrading::features::{FeaturePosition, RiskLimits};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Generate market events for benchmarking
fn generate_market_events(num_events: usize) -> Vec<MarketEvent> {
    let mut events = Vec::with_capacity(num_events);
    let mut timestamp = 1_000_000;
    let mut price = 100_000_000;
    
    for i in 0..num_events {
        timestamp += 100;
        price += ((i % 20) as i64) - 10;
        
        if i % 2 == 0 {
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
            events.push(MarketEvent::BBO(BBOUpdate {
                instrument_id: 1,
                bid_price: Some(Price::new(price - 50)),
                ask_price: Some(Price::new(price + 50)),
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

/// Create strategy context efficiently
fn create_strategy_context() -> StrategyContext {
    StrategyContext::new(
        "benchmark".to_string(),
        1_000_000,
        FeaturePosition::default(),
        RiskLimits::default(),
        true,
    )
}

/// Benchmark 1: Raw event iteration (baseline)
fn benchmark_raw_event_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_event_iteration");
    group.measurement_time(Duration::from_secs(2));
    
    for num_events in [100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_events),
            num_events,
            |b, &num_events| {
                let events = generate_market_events(num_events);
                
                b.iter(|| {
                    let start = Instant::now();
                    
                    for event in &events {
                        black_box(event);
                    }
                    
                    let elapsed = start.elapsed();
                    let throughput = events.len() as f64 / elapsed.as_secs_f64();
                    
                    if num_events == 1_000_000 {
                        println!("\nRaw iteration: {:.0} events/second", throughput);
                    }
                    
                    black_box(throughput);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark 2: Event processing with cloning (current bottleneck)
fn benchmark_event_cloning_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_cloning");
    group.measurement_time(Duration::from_secs(2));
    
    let num_events = 100_000;
    let events = generate_market_events(num_events);
    
    group.bench_function("with_cloning", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut cloned_events = Vec::with_capacity(events.len());
            
            for event in &events {
                cloned_events.push(event.clone()); // Simulate current cloning overhead
                black_box(&cloned_events.last());
            }
            
            let elapsed = start.elapsed();
            let throughput = events.len() as f64 / elapsed.as_secs_f64();
            
            println!("\nWith cloning: {:.0} events/second", throughput);
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
            
            println!("\nWithout cloning: {:.0} events/second", throughput);
            black_box(throughput);
        });
    });
    
    group.finish();
}

/// Benchmark 3: Strategy context creation overhead
fn benchmark_context_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_creation");
    group.measurement_time(Duration::from_secs(2));
    
    let num_iterations = 100_000;
    
    group.bench_function("context_creation", |b| {
        b.iter(|| {
            let start = Instant::now();
            
            for i in 0..num_iterations {
                let context = StrategyContext::new(
                    format!("strategy_{}", i),
                    1_000_000 + i as u64,
                    FeaturePosition::default(),
                    RiskLimits::default(),
                    true,
                );
                black_box(context);
            }
            
            let elapsed = start.elapsed();
            let throughput = num_iterations as f64 / elapsed.as_secs_f64();
            
            println!("\nContext creation: {:.0} contexts/second", throughput);
            black_box(throughput);
        });
    });
    
    group.finish();
}

/// Benchmark 4: String allocation overhead (features)
fn benchmark_string_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_allocations");
    group.measurement_time(Duration::from_secs(2));
    
    let num_operations = 100_000;
    
    group.bench_function("hashmap_with_strings", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut features: HashMap<String, f64> = HashMap::new();
            
            for i in 0..num_operations {
                features.insert(format!("feature_{}", i % 10), i as f64);
                features.insert("spread".to_string(), 0.05);
                features.insert("volume".to_string(), 1000.0);
            }
            
            let elapsed = start.elapsed();
            let throughput = (num_operations * 3) as f64 / elapsed.as_secs_f64();
            
            println!("\nString HashMap: {:.0} ops/second", throughput);
            black_box(throughput);
        });
    });
    
    group.bench_function("array_with_indices", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut features = [0.0f64; 10];
            
            for i in 0..num_operations {
                features[i % 10] = i as f64;
                features[0] = 0.05; // spread
                features[1] = 1000.0; // volume
            }
            
            let elapsed = start.elapsed();
            let throughput = (num_operations * 3) as f64 / elapsed.as_secs_f64();
            
            println!("\nArray indexing: {:.0} ops/second", throughput);
            black_box(throughput);
        });
    });
    
    group.finish();
}

/// Benchmark 5: Lock contention simulation
fn benchmark_lock_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("lock_contention");
    group.measurement_time(Duration::from_secs(2));
    
    use std::sync::{Arc, RwLock};
    
    let num_operations = 10_000;
    let data = Arc::new(RwLock::new(HashMap::<u32, f64>::new()));
    
    group.bench_function("with_locks", |b| {
        b.iter(|| {
            let start = Instant::now();
            
            for i in 0..num_operations {
                {
                    let mut map = data.write().unwrap();
                    map.insert(i % 100, i as f64);
                }
                {
                    let map = data.read().unwrap();
                    black_box(map.get(&(i % 100)));
                }
            }
            
            let elapsed = start.elapsed();
            let throughput = (num_operations * 2) as f64 / elapsed.as_secs_f64();
            
            println!("\nWith locks: {:.0} ops/second", throughput);
            black_box(throughput);
        });
    });
    
    group.bench_function("without_locks", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut local_data = HashMap::<u32, f64>::new();
            
            for i in 0..num_operations {
                local_data.insert(i % 100, i as f64);
                black_box(local_data.get(&(i % 100)));
            }
            
            let elapsed = start.elapsed();
            let throughput = (num_operations * 2) as f64 / elapsed.as_secs_f64();
            
            println!("\nWithout locks: {:.0} ops/second", throughput);
            black_box(throughput);
        });
    });
    
    group.finish();
}

/// Benchmark 6: Optimized strategy processing
fn benchmark_optimized_strategy_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimized_strategy");
    group.measurement_time(Duration::from_secs(3));
    
    for num_events in [10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_events),
            num_events,
            |b, &num_events| {
                let events = generate_market_events(num_events);
                
                b.iter(|| {
                    // Create strategy once (amortize creation cost)
                    let mut strategy = MeanReversionStrategy::new(
                        "OptimizedMR".to_string(),
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
                    
                    // Create context once (reuse)
                    let context = create_strategy_context();
                    
                    let start = Instant::now();
                    
                    // Process events with minimal overhead
                    for event in &events {
                        let _output = strategy.on_market_event(event, &context);
                        // Note: Not processing orders to focus on pure strategy overhead
                    }
                    
                    let elapsed = start.elapsed();
                    let throughput = events.len() as f64 / elapsed.as_secs_f64();
                    
                    if num_events == 100_000 {
                        println!("\nOptimized strategy: {:.0} events/second", throughput);
                    }
                    
                    black_box(throughput);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark 7: Performance target test
fn benchmark_performance_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_target");
    group.measurement_time(Duration::from_secs(5));
    
    // Test with different batch sizes to find optimal processing
    for batch_size in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let events = generate_market_events(batch_size);
                
                b.iter(|| {
                    // Simulate optimized processing pipeline
                    let mut strategy = MeanReversionStrategy::new(
                        "TargetTest".to_string(),
                        1,
                        MeanReversionConfig::default(),
                    );
                    
                    let context = create_strategy_context();
                    let start = Instant::now();
                    
                    // Batch process events
                    let chunk_size = 1000; // Process in chunks for cache efficiency
                    for chunk in events.chunks(chunk_size) {
                        for event in chunk {
                            let _output = strategy.on_market_event(event, &context);
                        }
                    }
                    
                    let elapsed = start.elapsed();
                    let throughput = events.len() as f64 / elapsed.as_secs_f64();
                    let efficiency = (throughput / 18_000_000.0) * 100.0;
                    
                    if batch_size == 100_000 {
                        println!("\nBatch processing: {:.0} events/second ({:.1}% of 18M target)", 
                                throughput, efficiency);
                    }
                    
                    black_box(throughput);
                });
            },
        );
    }
    
    group.finish();
}

/// Summary benchmark to print performance analysis
fn benchmark_performance_summary(c: &mut Criterion) {
    println!("\n=== Performance Bottleneck Analysis ===");
    println!("Current: 3.1M events/s (17% of 18M target)");
    println!("Target: >15M events/s (>83% of 18M target)");
    println!("\nExpected bottlenecks:");
    println!("1. Event cloning: 2-3x overhead");
    println!("2. Lock contention: 1.5-2x overhead");
    println!("3. String allocations: 1.5x overhead");
    println!("4. Context creation: Minor overhead");
    println!("\nRun individual benchmarks to see actual measurements.");
    
    let mut group = c.benchmark_group("summary");
    group.bench_function("performance_analysis", |b| {
        b.iter(|| {
            black_box(18_000_000); // Target throughput
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    benchmark_raw_event_iteration,
    benchmark_event_cloning_overhead,
    benchmark_context_overhead,
    benchmark_string_allocations,
    benchmark_lock_contention,
    benchmark_optimized_strategy_processing,
    benchmark_performance_target,
    benchmark_performance_summary
);
criterion_main!(benches);