//! Realistic performance test using actual market data files
//!
//! This test processes real MBO data to measure true throughput

use algotrading::backtest::{BacktestConfig, BacktestEngine};
use algotrading::core::MarketDataSource;
use algotrading::market_data::{FileReader, ZeroCopyFileReader};
use algotrading::strategies::MeanReversionStrategy;
use algotrading::strategies::mean_reversion::MeanReversionConfig;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    println!("=== REALISTIC PERFORMANCE TEST ===");
    println!("Testing with actual market data files");
    println!("Target: >15M events/s (>83% of 18M target)");
    println!("Previous baseline: 3.1M events/s\n");

    // Get market data files
    let data_dir = PathBuf::from("../Market_Data/GLBX-20250528-84NHYCGUFY");

    // Test with first file
    let test_file = data_dir.join("glbx-mdp3-20250428.mbo.dbn.zst");

    if !test_file.exists() {
        eprintln!("Error: Market data file not found at {:?}", test_file);
        eprintln!("Please ensure Market_Data directory is in the parent directory");
        return;
    }

    println!("Testing with file: {:?}\n", test_file.file_name().unwrap());

    // Test 1: Original FileReader performance
    println!("1. Testing ORIGINAL FileReader (baseline):");
    let original_throughput = test_original_reader(&test_file);

    // Test 2: Zero-copy FileReader performance
    println!("\n2. Testing ZERO-COPY FileReader:");
    let zerocopy_throughput = test_zerocopy_reader(&test_file);

    // Test 3: Full backtest pipeline (original)
    println!("\n3. Testing FULL BACKTEST (original):");
    let backtest_throughput = test_full_backtest(&test_file);

    // Test 4: Count actual events in file
    println!("\n4. Counting events in file:");
    let event_count = count_events_in_file(&test_file);

    // Summary
    println!("\n=== PERFORMANCE SUMMARY ===");
    println!("File contains: {} events", event_count);
    println!("Original FileReader: {:.0} events/s", original_throughput);
    println!(
        "Zero-copy FileReader: {:.0} events/s ({:.1}x improvement)",
        zerocopy_throughput,
        zerocopy_throughput / original_throughput
    );
    println!("Full backtest: {:.0} events/s", backtest_throughput);

    let efficiency = (zerocopy_throughput / 18_000_000.0) * 100.0;
    println!("\nZero-copy efficiency: {:.1}% of 18M target", efficiency);

    if zerocopy_throughput >= 15_000_000.0 {
        println!("✅ SUCCESS: Achieved target performance!");
    } else if zerocopy_throughput >= 10_000_000.0 {
        println!("✅ GOOD: Significant improvement");
    } else if zerocopy_throughput >= 6_000_000.0 {
        println!("⚠️  MODERATE: Some improvement");
    } else {
        println!("❌ More optimization needed");
    }
}

fn test_original_reader(file_path: &PathBuf) -> f64 {
    let start = Instant::now();
    let mut event_count = 0;

    let mut reader = FileReader::new(vec![file_path.clone()]).unwrap();

    let processing_start = Instant::now();
    while let Some(_event) = reader.next_update() {
        event_count += 1;
        // Don't do any processing to measure pure read speed
    }
    let processing_time = processing_start.elapsed();

    let total_time = start.elapsed();
    println!(
        "  Read {} events in {:.2}s",
        event_count,
        processing_time.as_secs_f64()
    );
    println!("  Total time with setup: {:.2}s", total_time.as_secs_f64());

    event_count as f64 / processing_time.as_secs_f64()
}

fn test_zerocopy_reader(file_path: &PathBuf) -> f64 {
    let start = Instant::now();
    let mut event_count = 0;

    let mut reader = ZeroCopyFileReader::new(vec![file_path.clone()]).unwrap();

    let processing_start = Instant::now();
    while let Some(_event) = reader.next_update() {
        event_count += 1;
        // Don't do any processing to measure pure read speed
    }
    let processing_time = processing_start.elapsed();

    let total_time = start.elapsed();
    println!(
        "  Read {} events in {:.2}s",
        event_count,
        processing_time.as_secs_f64()
    );
    println!("  Total time with setup: {:.2}s", total_time.as_secs_f64());

    event_count as f64 / processing_time.as_secs_f64()
}

fn test_full_backtest(file_path: &PathBuf) -> f64 {
    let start = Instant::now();

    // Create backtest engine with simple config
    let config = BacktestConfig::default();

    let mut engine = BacktestEngine::new(config);

    // Add a simple strategy
    let strategy = MeanReversionStrategy::new(
        "TestMR".to_string(),
        5921, // MES instrument
        MeanReversionConfig {
            lookback_period: 20,
            entry_threshold: 2.0,
            exit_threshold: 0.5,
            max_position_size: 10,
            order_size: 1,
            use_limit_orders: false,
            limit_order_offset_ticks: 1,
        },
    );

    engine.add_strategy(Box::new(strategy));

    // Run backtest and measure time
    let processing_start = Instant::now();
    let report = engine.run(&[file_path.clone()]).unwrap();
    let processing_time = processing_start.elapsed();

    let total_time = start.elapsed();
    let event_count = report.events_processed;

    println!(
        "  Processed {} events in {:.2}s",
        event_count,
        processing_time.as_secs_f64()
    );
    println!("  Total time with setup: {:.2}s", total_time.as_secs_f64());
    println!(
        "  Trades executed: {}",
        report.performance_metrics.total_trades
    );

    event_count as f64 / processing_time.as_secs_f64()
}

fn count_events_in_file(file_path: &PathBuf) -> usize {
    let mut reader = FileReader::new(vec![file_path.clone()]).unwrap();
    let mut count = 0;

    while let Some(_) = reader.next_update() {
        count += 1;
    }

    count
}
