//! Test ultra-fast backtesting engine
//!
//! Minimal features for maximum throughput

use algotrading::backtest::engine_ultra_fast::UltraFastEngine;
use algotrading::backtest::{BacktestConfig, BacktestEngine};
use algotrading::strategies::MeanReversionStrategy;
use algotrading::strategies::mean_reversion::MeanReversionConfig;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    println!("=== ULTRA-FAST BACKTESTING TEST ===");
    println!("Target: >15M events/s (>83% of 18M)\n");

    // Get data files
    let data_dir = PathBuf::from("../Market_Data/GLBX-20250528-84NHYCGUFY");
    let test_file = data_dir.join("glbx-mdp3-20250428.mbo.dbn.zst");

    if !test_file.exists() {
        eprintln!("Error: Market data file not found");
        return;
    }

    let files = vec![test_file];

    // Test 1: Original engine (baseline)
    println!("1. ORIGINAL Engine:");
    let original = test_original(&files);

    // Test 2: Ultra-fast engine
    println!("\n2. ULTRA-FAST Engine:");
    let ultra_fast = test_ultra_fast(&files);

    // Compare
    println!("\n=== COMPARISON ===");
    println!(
        "Original: {:.0} events/s ({:.1}% of 18M)",
        original,
        (original / 18_000_000.0) * 100.0
    );
    println!(
        "Ultra-fast: {:.0} events/s ({:.1}% of 18M)",
        ultra_fast,
        (ultra_fast / 18_000_000.0) * 100.0
    );

    if original > 0.0 {
        println!("Improvement: {:.1}x", ultra_fast / original);
    }
}

fn test_original(files: &[PathBuf]) -> f64 {
    let start = Instant::now();

    let config = BacktestConfig::default();
    let mut engine = BacktestEngine::new(config);

    let strategy =
        MeanReversionStrategy::new("Test".to_string(), 5921, MeanReversionConfig::default());

    let _ = engine.add_strategy(Box::new(strategy));

    match engine.run(files) {
        Ok(report) => {
            let elapsed = start.elapsed();
            println!("  Events: {}", report.events_processed);
            println!("  Time: {:.2}s", elapsed.as_secs_f64());
            report.events_processed as f64 / elapsed.as_secs_f64()
        }
        Err(e) => {
            eprintln!("  Error: {}", e);
            0.0
        }
    }
}

fn test_ultra_fast(files: &[PathBuf]) -> f64 {
    let mut engine = UltraFastEngine::new();

    let strategy =
        MeanReversionStrategy::new("Test".to_string(), 5921, MeanReversionConfig::default());

    match engine.add_strategy(Box::new(strategy)) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("  Error: {}", e);
            return 0.0;
        }
    }

    match engine.run(files) {
        Ok(report) => {
            report.print_summary();
            report.throughput
        }
        Err(e) => {
            eprintln!("  Error: {}", e);
            0.0
        }
    }
}
