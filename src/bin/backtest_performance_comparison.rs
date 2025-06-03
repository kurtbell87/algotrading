//! Performance comparison between original and optimized backtest engines
//!
//! Tests with real MBO data files

use algotrading::backtest::engine_fast::{FastBacktestConfig, FastBacktestEngine};
use algotrading::backtest::{BacktestConfig, BacktestEngine};
use algotrading::strategies::MeanReversionStrategy;
use algotrading::strategies::mean_reversion::MeanReversionConfig;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    println!("=== BACKTEST PERFORMANCE COMPARISON ===");
    println!("Testing with real market data files");
    println!("Target: >15M events/s (>83% of 18M target)\n");

    // Get market data files
    let data_dir = PathBuf::from("../Market_Data/GLBX-20250528-84NHYCGUFY");
    let test_files = vec![
        data_dir.join("glbx-mdp3-20250428.mbo.dbn.zst"),
        // Add more files for larger test
        // data_dir.join("glbx-mdp3-20250429.mbo.dbn.zst"),
        // data_dir.join("glbx-mdp3-20250430.mbo.dbn.zst"),
    ];

    // Verify files exist
    for file in &test_files {
        if !file.exists() {
            eprintln!("Error: Market data file not found at {:?}", file);
            return;
        }
    }

    // Create test strategy
    let strategy_config = MeanReversionConfig {
        lookback_period: 20,
        entry_threshold: 2.0,
        exit_threshold: 0.5,
        max_position_size: 10,
        order_size: 1,
        use_limit_orders: false,
        limit_order_offset_ticks: 1,
    };

    // Test 1: Original BacktestEngine
    println!("1. Testing ORIGINAL BacktestEngine:");
    let original_throughput = test_original_engine(&test_files, &strategy_config);

    // Test 2: Fast BacktestEngine
    println!("\n2. Testing FAST BacktestEngine:");
    let fast_throughput = test_fast_engine(&test_files, &strategy_config);

    // Summary
    println!("\n=== PERFORMANCE SUMMARY ===");
    println!("Original engine: {:.0} events/s", original_throughput);
    println!(
        "Fast engine: {:.0} events/s ({:.1}x improvement)",
        fast_throughput,
        fast_throughput / original_throughput
    );

    let original_efficiency = (original_throughput / 18_000_000.0) * 100.0;
    let fast_efficiency = (fast_throughput / 18_000_000.0) * 100.0;

    println!("\nEfficiency vs 18M target:");
    println!("Original: {:.1}%", original_efficiency);
    println!("Fast: {:.1}%", fast_efficiency);

    if fast_throughput >= 15_000_000.0 {
        println!("\n✅ SUCCESS: Fast engine achieved target performance!");
    } else if fast_throughput >= 10_000_000.0 {
        println!("\n✅ GOOD: Significant improvement achieved");
    } else if fast_throughput >= 6_000_000.0 {
        println!("\n⚠️  MODERATE: Some improvement, but more optimization needed");
    } else {
        println!("\n❌ NEEDS WORK: Performance still below acceptable threshold");
    }
}

fn test_original_engine(files: &[PathBuf], strategy_config: &MeanReversionConfig) -> f64 {
    let start = Instant::now();

    // Create engine with default config
    let config = BacktestConfig::default();
    let mut engine = BacktestEngine::new(config);

    // Add strategy
    let strategy = MeanReversionStrategy::new(
        "TestMR".to_string(),
        5921, // MES instrument
        strategy_config.clone(),
    );

    let _ = engine.add_strategy(Box::new(strategy));

    // Run backtest
    let processing_start = Instant::now();
    match engine.run(files) {
        Ok(report) => {
            let processing_time = processing_start.elapsed();
            let total_time = start.elapsed();

            println!("  Events processed: {}", report.events_processed);
            println!("  Processing time: {:.2}s", processing_time.as_secs_f64());
            println!("  Total time: {:.2}s", total_time.as_secs_f64());
            println!("  Trades: {}", report.performance_metrics.total_trades);

            report.events_processed as f64 / processing_time.as_secs_f64()
        }
        Err(e) => {
            eprintln!("  Error: {}", e);
            0.0
        }
    }
}

fn test_fast_engine(files: &[PathBuf], strategy_config: &MeanReversionConfig) -> f64 {
    let start = Instant::now();

    // Create fast engine
    let config = FastBacktestConfig::default();
    let mut engine = FastBacktestEngine::new(config);

    // Add strategy
    let strategy = MeanReversionStrategy::new(
        "TestMR".to_string(),
        5921, // MES instrument
        strategy_config.clone(),
    );

    let _ = engine.add_strategy(Box::new(strategy));

    // Run backtest
    let processing_start = Instant::now();
    match engine.run(files) {
        Ok(report) => {
            let _processing_time = processing_start.elapsed();
            let total_time = start.elapsed();

            report.print_summary();
            println!("  Total time with setup: {:.2}s", total_time.as_secs_f64());

            report.throughput
        }
        Err(e) => {
            eprintln!("  Error: {}", e);
            0.0
        }
    }
}
