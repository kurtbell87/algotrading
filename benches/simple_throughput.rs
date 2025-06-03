//! Simple throughput benchmark to compare against 18M msg/s target
//!
//! This benchmark creates temporary market data files and measures
//! the throughput of the backtesting engine.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use algotrading::core::types::{InstrumentId, Price, Quantity};
use algotrading::core::{Side, MarketUpdate, Trade, BBO};
use algotrading::backtest::engine::{BacktestEngine, BacktestConfig};
use algotrading::backtest::execution::{LatencyModel, FillModel};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

/// Generate synthetic market data and write to temporary DBN file
fn create_test_data_file(num_events: usize, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create directory if it doesn't exist
    if let Some(parent) = std::path::Path::new(file_path).parent() {
        create_dir_all(parent)?;
    }
    
    // For this benchmark, we'll create a simple text file with mock data
    // In practice, this would be DBN format
    let mut file = File::create(file_path)?;
    
    let mut timestamp = 1_000_000;
    let mut price = 100_000_000; // 100.0 in fixed point
    
    for i in 0..num_events {
        timestamp += 1000; // 1ms between events
        
        // Simulate price movement
        price += ((i % 20) as i64) - 10;
        
        // Write a simple line format that our mock reader can parse
        if i % 2 == 0 {
            // Trade
            writeln!(file, "TRADE,{},{},{},{}", timestamp, price, 100 + (i % 100), if i % 3 == 0 { "BID" } else { "ASK" })?;
        } else {
            // BBO
            let spread = 100;
            writeln!(file, "BBO,{},{},{},{},{}", timestamp, price - spread/2, price + spread/2, 200, 200)?;
        }
    }
    
    Ok(())
}

fn benchmark_pure_lob_replay(c: &mut Criterion) {
    let mut group = c.benchmark_group("pure_lob_replay");
    group.measurement_time(Duration::from_secs(10));
    
    for num_events in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));
        
        group.bench_with_input(
            BenchmarkId::new("events", num_events),
            num_events,
            |b, &num_events| {
                // Create test data file
                let file_path = format!("/tmp/benchmark_data_{}.dbn", num_events);
                create_test_data_file(num_events, &file_path).expect("Failed to create test data");
                
                b.iter(|| {
                    // Simulate pure LOB replay (no strategies)
                    let config = BacktestConfig {
                        initial_capital: 100_000.0,
                        calculate_features: false,
                        latency_model: LatencyModel::Zero,
                        fill_model: FillModel::Optimistic,
                        ..Default::default()
                    };
                    
                    let mut engine = BacktestEngine::new(config);
                    
                    // Run without strategies for pure throughput
                    let report = engine.run(&[&file_path]).expect("Backtest failed");
                    black_box(report.events_processed);
                });
                
                // Cleanup
                let _ = std::fs::remove_file(&file_path);
            },
        );
    }
    
    group.finish();
}

fn benchmark_single_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_strategy");
    group.measurement_time(Duration::from_secs(10));
    
    for num_events in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));
        
        group.bench_with_input(
            BenchmarkId::new("mean_reversion", num_events),
            num_events,
            |b, &num_events| {
                let file_path = format!("/tmp/benchmark_strategy_{}.dbn", num_events);
                create_test_data_file(num_events, &file_path).expect("Failed to create test data");
                
                b.iter(|| {
                    let config = BacktestConfig {
                        initial_capital: 100_000.0,
                        calculate_features: false,
                        latency_model: LatencyModel::Fixed(100),
                        fill_model: FillModel::Optimistic,
                        ..Default::default()
                    };
                    
                    let mut engine = BacktestEngine::new(config);
                    
                    // Add mean reversion strategy
                    let strategy = MeanReversionStrategy::new(
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
                    
                    engine.add_strategy(Box::new(strategy)).expect("Failed to add strategy");
                    
                    let report = engine.run(&[&file_path]).expect("Backtest failed");
                    black_box(report.events_processed);
                });
                
                let _ = std::fs::remove_file(&file_path);
            },
        );
    }
    
    group.finish();
}

fn benchmark_multiple_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_strategies");
    group.measurement_time(Duration::from_secs(15));
    
    let num_events = 50_000;
    let file_path = format!("/tmp/benchmark_multi_{}.dbn", num_events);
    create_test_data_file(num_events, &file_path).expect("Failed to create test data");
    
    for num_strategies in [1, 2, 3, 5].iter() {
        group.throughput(Throughput::Elements(num_events as u64));
        
        group.bench_with_input(
            BenchmarkId::new("strategies", num_strategies),
            num_strategies,
            |b, &num_strategies| {
                b.iter(|| {
                    let config = BacktestConfig {
                        initial_capital: 100_000.0,
                        calculate_features: false,
                        latency_model: LatencyModel::Fixed(100),
                        fill_model: FillModel::Optimistic,
                        ..Default::default()
                    };
                    
                    let mut engine = BacktestEngine::new(config);
                    
                    // Add multiple strategies
                    for i in 0..num_strategies {
                        let strategy = MeanReversionStrategy::new(
                            format!("BenchmarkMR_{}", i),
                            1,
                            MeanReversionConfig {
                                lookback_period: 20 + (i * 10) as usize,
                                entry_threshold: 1.5,
                                exit_threshold: 0.5,
                                max_position_size: 5,
                                order_size: 1,
                                use_limit_orders: false,
                                limit_order_offset_ticks: 1,
                            },
                        );
                        
                        engine.add_strategy(Box::new(strategy)).expect("Failed to add strategy");
                    }
                    
                    let report = engine.run(&[&file_path]).expect("Backtest failed");
                    black_box(report.events_processed);
                });
            },
        );
    }
    
    // Cleanup
    let _ = std::fs::remove_file(&file_path);
    group.finish();
}

fn benchmark_feature_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");
    group.measurement_time(Duration::from_secs(10));
    
    let num_events = 10_000;
    let file_path = format!("/tmp/benchmark_features_{}.dbn", num_events);
    create_test_data_file(num_events, &file_path).expect("Failed to create test data");
    
    for features_enabled in [false, true].iter() {
        let label = if *features_enabled { "with_features" } else { "without_features" };
        
        group.bench_function(label, |b| {
            b.iter(|| {
                let config = BacktestConfig {
                    initial_capital: 100_000.0,
                    calculate_features: *features_enabled,
                    latency_model: LatencyModel::Fixed(100),
                    fill_model: FillModel::Optimistic,
                    ..Default::default()
                };
                
                let mut engine = BacktestEngine::new(config);
                
                let strategy = MeanReversionStrategy::new(
                    "BenchmarkMR".to_string(),
                    1,
                    MeanReversionConfig::default(),
                );
                
                engine.add_strategy(Box::new(strategy)).expect("Failed to add strategy");
                
                let report = engine.run(&[&file_path]).expect("Backtest failed");
                black_box(report.events_processed);
            });
        });
    }
    
    // Cleanup
    let _ = std::fs::remove_file(&file_path);
    group.finish();
}

criterion_group!(
    benches,
    benchmark_pure_lob_replay,
    benchmark_single_strategy,
    benchmark_multiple_strategies,
    benchmark_feature_overhead
);
criterion_main!(benches);