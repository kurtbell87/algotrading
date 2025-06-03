//! Simple performance test for backtesting optimization
//!
//! Uses real MBO data files

use algotrading::backtest::engine_simple_fast::{SimpleFastEngine, print_performance_comparison};
use algotrading::backtest::{BacktestConfig, BacktestEngine};
use algotrading::strategies::MeanReversionStrategy;
use algotrading::strategies::mean_reversion::MeanReversionConfig;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    println!("=== SIMPLE BACKTEST PERFORMANCE TEST ===");
    println!("Target: >15M events/s (>83% of 18M target)");
    println!("Previous baseline: 3.6M events/s\n");

    // Get market data files
    let data_dir = PathBuf::from("../Market_Data/GLBX-20250528-84NHYCGUFY");
    let test_file = data_dir.join("glbx-mdp3-20250428.mbo.dbn.zst");

    if !test_file.exists() {
        eprintln!("Error: Market data file not found at {:?}", test_file);
        return;
    }

    // Test 1: Original BacktestEngine
    println!("1. Testing ORIGINAL BacktestEngine:");
    let original_throughput = test_original(&test_file);

    // Test 2: Simple Fast Engine
    println!("\n2. Testing SIMPLE FAST BacktestEngine:");
    let fast_throughput = test_simple_fast(&test_file);

    // Print comparison
    print_performance_comparison(original_throughput, fast_throughput);
}

fn test_original(file: &PathBuf) -> f64 {
    let _start = Instant::now();

    // Create engine
    let config = BacktestConfig::default();
    let mut engine = BacktestEngine::new(config);

    // Add strategy
    let strategy =
        MeanReversionStrategy::new("TestMR".to_string(), 5921, MeanReversionConfig::default());

    let _ = engine.add_strategy(Box::new(strategy));

    // Run backtest
    let processing_start = Instant::now();
    match engine.run(&[file]) {
        Ok(report) => {
            let processing_time = processing_start.elapsed();

            println!("  Events: {}", report.events_processed);
            println!("  Time: {:.2}s", processing_time.as_secs_f64());

            report.events_processed as f64 / processing_time.as_secs_f64()
        }
        Err(e) => {
            eprintln!("  Error: {}", e);
            0.0
        }
    }
}

fn test_simple_fast(file: &PathBuf) -> f64 {
    let _start = Instant::now();

    // Create fast engine
    let config = BacktestConfig::default();
    let mut engine = SimpleFastEngine::new(config);

    // Add strategy
    let strategy =
        MeanReversionStrategy::new("TestMR".to_string(), 5921, MeanReversionConfig::default());

    let _ = engine.add_strategy(Box::new(strategy));

    // Run backtest
    let processing_start = Instant::now();
    match engine.run(&[file]) {
        Ok(report) => {
            let processing_time = processing_start.elapsed();

            println!("  Events: {}", report.events_processed);
            println!("  Time: {:.2}s", processing_time.as_secs_f64());

            report.events_processed as f64 / processing_time.as_secs_f64()
        }
        Err(e) => {
            eprintln!("  Error: {}", e);
            0.0
        }
    }
}
