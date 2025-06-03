//! Optimized throughput benchmark
//!
//! Tests the performance improvements from:
//! 1. Zero-copy event processing
//! 2. Lockless market snapshots
//! 3. Indexed feature system
//! 4. Pre-allocated buffers

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use algotrading::core::types::{InstrumentId, Price, Quantity};
use algotrading::core::{Side, MarketUpdate, Trade, BBO};
use algotrading::market_data::reader_optimized::{MarketSnapshot, IndexedFeatureVector, feature_indices};
use algotrading::strategy::{Strategy, StrategyConfig, StrategyContext, StrategyOutput, StrategyError};
use algotrading::features::{FeaturePosition, RiskLimits};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use std::time::{Duration, Instant};

/// Generate test market updates (optimized format)
fn generate_market_updates(num_events: usize) -> Vec<MarketUpdate> {
    let mut updates = Vec::with_capacity(num_events);
    let mut timestamp = 1_000_000;
    let mut price = 100_000_000; // 100.00 in fixed point
    
    for i in 0..num_events {
        timestamp += 100; // 100 microseconds between events
        price += ((i % 20) as i64) - 10;
        
        if i % 2 == 0 {
            // Trade update
            updates.push(MarketUpdate::Trade(Trade {
                instrument_id: 1,
                price: Price::new(price),
                quantity: Quantity::from((100 + (i % 100)) as u32),
                side: if i % 3 == 0 { Side::Bid } else { Side::Ask },
                timestamp,
            }));
        } else {
            // BBO update
            let spread = 100;
            updates.push(MarketUpdate::BBO(BBO {
                instrument_id: 1,
                bid_price: Price::new(price - spread / 2),
                ask_price: Price::new(price + spread / 2),
                bid_quantity: Quantity::from(200u32),
                ask_quantity: Quantity::from(200u32),
                timestamp,
            }));
        }
    }
    
    updates
}

/// Benchmark pure market update processing (no conversion overhead)
fn benchmark_market_update_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimized_market_updates");
    group.measurement_time(Duration::from_secs(3));
    
    for num_events in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_events),
            num_events,
            |b, &num_events| {
                let updates = generate_market_updates(num_events);
                
                b.iter(|| {
                    let start = Instant::now();
                    
                    // Pure processing without cloning
                    for update in &updates {
                        black_box(update);
                        let _timestamp = update.timestamp();
                        let _instrument = update.instrument_id();
                    }
                    
                    let elapsed = start.elapsed();
                    let throughput = updates.len() as f64 / elapsed.as_secs_f64();
                    
                    if num_events == 1_000_000 {
                        println!("\nOptimized market updates: {:.0} events/second", throughput);
                    }
                    
                    black_box(throughput);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark lockless market snapshot operations
fn benchmark_lockless_market_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("lockless_snapshot");
    group.measurement_time(Duration::from_secs(3));
    
    let num_events = 100_000;
    let updates = generate_market_updates(num_events);
    
    group.bench_function("snapshot_updates", |b| {
        b.iter(|| {
            let mut snapshot = MarketSnapshot::new();
            let start = Instant::now();
            
            for update in &updates {
                match update {
                    MarketUpdate::Trade(trade) => {
                        snapshot.update_last_price(trade.instrument_id, trade.price, trade.timestamp);
                    }
                    MarketUpdate::BBO(bbo) => {
                        snapshot.update_bbo(
                            bbo.instrument_id,
                            bbo.bid_price,
                            bbo.ask_price,
                            bbo.bid_quantity,
                            bbo.ask_quantity,
                            bbo.timestamp,
                        );
                    }
                    _ => {}
                }
            }
            
            let elapsed = start.elapsed();
            let throughput = updates.len() as f64 / elapsed.as_secs_f64();
            
            println!("\nLockless snapshot: {:.0} updates/second", throughput);
            black_box(throughput);
        });
    });
    
    group.finish();
}

/// Benchmark indexed feature system vs string-based
fn benchmark_indexed_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexed_features");
    group.measurement_time(Duration::from_secs(3));
    
    let num_updates = 50_000;
    let updates = generate_market_updates(num_updates);
    
    // Test indexed features
    group.bench_function("indexed_features", |b| {
        b.iter(|| {
            let mut features = IndexedFeatureVector::new(feature_indices::TOTAL_FEATURES, 0);
            let start = Instant::now();
            
            for update in &updates {
                if let MarketUpdate::BBO(bbo) = update {
                    let spread_abs = bbo.ask_price.0 - bbo.bid_price.0;
                    let spread_rel = spread_abs as f64 / bbo.bid_price.as_f64();
                    let imbalance = (bbo.bid_quantity.as_i64() - bbo.ask_quantity.as_i64()) as f64 
                        / (bbo.bid_quantity.as_i64() + bbo.ask_quantity.as_i64()) as f64;
                    
                    // Indexed operations (no string allocations)
                    features.set(feature_indices::SPREAD_ABSOLUTE, spread_abs as f64);
                    features.set(feature_indices::SPREAD_RELATIVE, spread_rel);
                    features.set(feature_indices::BID_SIZE, bbo.bid_quantity.as_f64());
                    features.set(feature_indices::ASK_SIZE, bbo.ask_quantity.as_f64());
                    features.set(feature_indices::VOLUME_IMBALANCE, imbalance);
                }
            }
            
            let elapsed = start.elapsed();
            let throughput = updates.len() as f64 / elapsed.as_secs_f64();
            
            println!("\nIndexed features: {:.0} updates/second", throughput);
            black_box(throughput);
        });
    });
    
    // Test string-based features for comparison
    group.bench_function("string_features", |b| {
        b.iter(|| {
            use std::collections::HashMap;
            let mut features: HashMap<String, f64> = HashMap::new();
            let start = Instant::now();
            
            for update in &updates {
                if let MarketUpdate::BBO(bbo) = update {
                    let spread_abs = bbo.ask_price.0 - bbo.bid_price.0;
                    let spread_rel = spread_abs as f64 / bbo.bid_price.as_f64();
                    let imbalance = (bbo.bid_quantity.as_i64() - bbo.ask_quantity.as_i64()) as f64 
                        / (bbo.bid_quantity.as_i64() + bbo.ask_quantity.as_i64()) as f64;
                    
                    // String-based operations (lots of allocations)
                    features.insert("spread_absolute".to_string(), spread_abs as f64);
                    features.insert("spread_relative".to_string(), spread_rel);
                    features.insert("bid_size".to_string(), bbo.bid_quantity.as_f64());
                    features.insert("ask_size".to_string(), bbo.ask_quantity.as_f64());
                    features.insert("volume_imbalance".to_string(), imbalance);
                }
            }
            
            let elapsed = start.elapsed();
            let throughput = updates.len() as f64 / elapsed.as_secs_f64();
            
            println!("\nString features: {:.0} updates/second", throughput);
            black_box(throughput);
        });
    });
    
    group.finish();
}

/// Benchmark optimized strategy processing
fn benchmark_optimized_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimized_strategy");
    group.measurement_time(Duration::from_secs(5));
    
    for num_events in [10_000, 100_000, 500_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_events),
            num_events,
            |b, &num_events| {
                let updates = generate_market_updates(num_events);
                
                b.iter(|| {
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
                    
                    // Use optimized context
                    let context = StrategyContext::new(
                        "optimized_mr".to_string(),
                        1_000_000,
                        FeaturePosition::default(),
                        RiskLimits::default(),
                        true,
                    );
                    
                    let start = Instant::now();
                    
                    // Process updates through strategy
                    for update in &updates {
                        // Convert to market event (minimal overhead)
                        let market_event = convert_update_to_event(update);
                        let _output = strategy.on_market_event(&market_event, &context);
                    }
                    
                    let elapsed = start.elapsed();
                    let throughput = updates.len() as f64 / elapsed.as_secs_f64();
                    
                    if num_events == 500_000 {
                        println!("\nOptimized strategy: {:.0} events/second", throughput);
                    }
                    
                    black_box(throughput);
                });
            },
        );
    }
    
    group.finish();
}

/// Convert MarketUpdate to MarketEvent efficiently
fn convert_update_to_event(update: &MarketUpdate) -> algotrading::market_data::events::MarketEvent {
    match update {
        MarketUpdate::Trade(trade) => {
            algotrading::market_data::events::MarketEvent::Trade(
                algotrading::market_data::events::TradeEvent {
                    instrument_id: trade.instrument_id,
                    trade_id: 0,
                    price: trade.price,
                    quantity: trade.quantity,
                    aggressor_side: trade.side,
                    timestamp: trade.timestamp,
                    buyer_order_id: None,
                    seller_order_id: None,
                }
            )
        }
        MarketUpdate::BBO(bbo) => {
            algotrading::market_data::events::MarketEvent::BBO(
                algotrading::market_data::events::BBOUpdate {
                    instrument_id: bbo.instrument_id,
                    bid_price: Some(bbo.bid_price),
                    ask_price: Some(bbo.ask_price),
                    bid_quantity: Some(bbo.bid_quantity),
                    ask_quantity: Some(bbo.ask_quantity),
                    bid_order_count: None,
                    ask_order_count: None,
                    timestamp: bbo.timestamp,
                }
            )
        }
        _ => {
            algotrading::market_data::events::MarketEvent::Trade(
                algotrading::market_data::events::TradeEvent {
                    instrument_id: 1,
                    trade_id: 0,
                    price: Price::new(100_000_000),
                    quantity: Quantity::from(100u32),
                    aggressor_side: Side::Bid,
                    timestamp: 1_000_000,
                    buyer_order_id: None,
                    seller_order_id: None,
                }
            )
        }
    }
}

/// Performance comparison summary
fn benchmark_performance_comparison(c: &mut Criterion) {
    println!("\n=== Performance Optimization Results ===");
    println!("Target: 18,000,000 events/second (pure LOB replay)");
    println!("\nOptimizations implemented:");
    println!("1. Zero-copy event processing");
    println!("2. Lockless market snapshots");
    println!("3. Indexed feature system");
    println!("4. Pre-allocated buffers");
    println!("\nExpected improvements:");
    println!("- Market updates: 150M+ events/s (baseline)");
    println!("- Lockless snapshots: 50M+ updates/s");
    println!("- Indexed features: 5-10x faster than strings");
    println!("- Strategy processing: 10-15M events/s target");
    
    // This benchmark just prints the summary
    let mut group = c.benchmark_group("summary");
    group.bench_function("performance_summary", |b| {
        b.iter(|| {
            black_box(());
        });
    });
    group.finish();
}

// Add trait implementations for benchmarking
impl MarketUpdate {
    pub fn timestamp(&self) -> u64 {
        match self {
            MarketUpdate::Trade(trade) => trade.timestamp,
            MarketUpdate::BBO(bbo) => bbo.timestamp,
            _ => 0,
        }
    }
    
    pub fn instrument_id(&self) -> InstrumentId {
        match self {
            MarketUpdate::Trade(trade) => trade.instrument_id,
            MarketUpdate::BBO(bbo) => bbo.instrument_id,
            _ => 0,
        }
    }
}

criterion_group!(
    benches,
    benchmark_market_update_processing,
    benchmark_lockless_market_snapshot,
    benchmark_indexed_features,
    benchmark_optimized_strategy,
    benchmark_performance_comparison
);
criterion_main!(benches);