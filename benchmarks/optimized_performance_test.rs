//! Test the optimized backtesting engine performance
//!
//! Compares original vs optimized implementations using real MBO data

use algotrading::backtest::engine_optimized_v2::OptimizedEngineV2;
use algotrading::backtest::{BacktestConfig, BacktestEngine};
use algotrading::strategies::MeanReversionStrategy;
use algotrading::strategies::mean_reversion::MeanReversionConfig;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    println!("=== OPTIMIZED BACKTESTING PERFORMANCE TEST ===");
    println!("Testing with real MBO data files");
    println!("Target: >15M events/s (>83% of 18M target)\n");

    // Get market data files
    let data_dir = PathBuf::from("../Market_Data/GLBX-20250528-84NHYCGUFY");
    let test_files = vec![data_dir.join("glbx-mdp3-20250428.mbo.dbn.zst")];

    // Verify files exist
    for file in &test_files {
        if !file.exists() {
            eprintln!("Error: Market data file not found at {:?}", file);
            return;
        }
    }

    // Create strategy config
    let strategy_config = MeanReversionConfig {
        lookback_period: 20,
        entry_threshold: 2.0,
        exit_threshold: 0.5,
        max_position_size: 10,
        order_size: 1,
        use_limit_orders: false,
        limit_order_offset_ticks: 1,
    };

    // Test 1: Original BacktestEngine (baseline)
    println!("1. Testing ORIGINAL BacktestEngine:");
    let original_throughput = test_original_engine(&test_files, &strategy_config);

    // Test 2: Optimized Engine V2
    println!("\n2. Testing OPTIMIZED Engine V2:");
    let optimized_throughput = test_optimized_engine(&test_files, &strategy_config);

    // Summary
    print_performance_summary(original_throughput, optimized_throughput);
}

fn test_original_engine(files: &[PathBuf], strategy_config: &MeanReversionConfig) -> f64 {
    let start = Instant::now();

    // Create engine
    let config = BacktestConfig::default();
    let mut engine = BacktestEngine::new(config);

    // Add strategy
    let strategy = MeanReversionStrategy::new(
        "TestMR".to_string(),
        5921, // MES
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

            let throughput = report.events_processed as f64 / processing_time.as_secs_f64();
            println!("  Throughput: {:.0} events/s", throughput);

            throughput
        }
        Err(e) => {
            eprintln!("  Error: {}", e);
            0.0
        }
    }
}

fn test_optimized_engine(files: &[PathBuf], strategy_config: &MeanReversionConfig) -> f64 {
    let start = Instant::now();

    // Create optimized engine
    let config = BacktestConfig::default();
    let mut engine = OptimizedEngineV2::new(config);

    // Add strategy
    let strategy = MeanReversionStrategy::new(
        "TestMR".to_string(),
        5921, // MES
        strategy_config.clone(),
    );

    match engine.add_strategy(Box::new(strategy)) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("  Error adding strategy: {}", e);
            return 0.0;
        }
    }

    // Run backtest
    match engine.run(files) {
        Ok(report) => {
            let total_time = start.elapsed();

            let throughput = report.events_processed as f64 / total_time.as_secs_f64();
            throughput
        }
        Err(e) => {
            eprintln!("  Error: {}", e);
            0.0
        }
    }
}

fn print_performance_summary(original: f64, optimized: f64) {
    println!("\n=== PERFORMANCE SUMMARY ===");
    println!("Original engine: {:.0} events/s", original);
    println!("Optimized engine: {:.0} events/s", optimized);

    if original > 0.0 {
        let improvement = optimized / original;
        println!("Improvement: {:.1}x", improvement);
    }

    let original_efficiency = (original / 18_000_000.0) * 100.0;
    let optimized_efficiency = (optimized / 18_000_000.0) * 100.0;

    println!("\nEfficiency vs 18M target:");
    println!("Original: {:.1}%", original_efficiency);
    println!("Optimized: {:.1}%", optimized_efficiency);

    if optimized >= 15_000_000.0 {
        println!("\n🎯 SUCCESS: Achieved target performance (>15M events/s)!");
    } else if optimized >= 10_000_000.0 {
        println!("\n✅ GOOD: Significant improvement (>10M events/s)");
    } else if optimized >= 6_000_000.0 {
        println!("\n⚠️  MODERATE: Some improvement (>6M events/s)");
    } else {
        println!("\n❌ NEEDS WORK: Still below acceptable threshold");
    }

    // Analysis
    println!("\nPerformance Analysis:");
    if optimized > original * 1.5 {
        println!("- Batch processing provided significant gains");
        println!("- Reduced lock contention improved throughput");
        println!("- Pre-allocated buffers reduced allocation overhead");
    }

    if optimized < 10_000_000.0 {
        println!("\nFurther optimizations to consider:");
        println!("- Eliminate event conversion overhead");
        println!("- Use lock-free data structures");
        println!("- Implement SIMD for mathematical operations");
        println!("- Profile and optimize hot paths");
    }
}
