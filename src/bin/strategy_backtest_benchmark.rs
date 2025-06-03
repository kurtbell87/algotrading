//! Strategy Backtest Performance Benchmark
//! 
//! This benchmark measures the ACTUAL performance of running a complete backtest
//! with a trading strategy on real MBO data files.

use algotrading::backtest::{BacktestConfig, BacktestEngine};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    println!("=== STRATEGY BACKTEST PERFORMANCE BENCHMARK ===");
    println!("Testing COMPLETE backtest with strategy execution on real MBO data\n");

    // Get the market data directory
    let data_dir = Path::new("../Market_Data/GLBX-20250528-84NHYCGUFY");
    if !data_dir.exists() {
        eprintln!("Error: Market data directory not found at {:?}", data_dir);
        return;
    }

    // Test with one full day of data
    let test_file = data_dir.join("glbx-mdp3-20250428.mbo.dbn.zst");
    if !test_file.exists() {
        eprintln!("Error: Test file not found at {:?}", test_file);
        return;
    }

    // Create mean reversion strategy
    let strategy = MeanReversionStrategy::new(
        "MR_Benchmark".to_string(),
        5921, // MES futures
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

    println!("Configuration:");
    println!("  Strategy: Mean Reversion");
    println!("  Lookback: 20 periods");
    println!("  Entry Threshold: 2.0 std devs");
    println!("  Max Position: 10 contracts");
    println!("  Data File: {}", test_file.file_name().unwrap().to_str().unwrap());

    // Create backtest engine with configuration
    println!("\nInitializing backtest engine...");
    let config = BacktestConfig::default();
    let mut engine = BacktestEngine::new(config);
    
    // Add the strategy
    engine.add_strategy(Box::new(strategy)).expect("Failed to add strategy");

    // Run the backtest
    println!("\nRunning backtest...");
    let backtest_start = Instant::now();
    
    // Run backtest on the file
    let report = engine.run(&[test_file.clone()]).expect("Failed to run backtest");
    
    let backtest_duration = backtest_start.elapsed();
    println!("Completed: {} events in {:.3}s", report.events_processed, backtest_duration.as_secs_f64());

    // Calculate throughput
    let msgs_per_sec = report.events_processed as f64 / backtest_duration.as_secs_f64();
    let nanos_per_msg = backtest_duration.as_nanos() as f64 / report.events_processed as f64;

    println!("\n=== PERFORMANCE RESULTS ===");
    println!("Events Processed: {}", report.events_processed);
    println!("Total Time: {:.3}s", backtest_duration.as_secs_f64());
    println!("Throughput: {:.2}M events/s", msgs_per_sec / 1_000_000.0);
    println!("Latency: {:.0} ns/event", nanos_per_msg);
    println!("vs 18M ceiling: {:.1}%", (msgs_per_sec / 18_000_000.0) * 100.0);
    println!("vs 3M baseline: {:.1}x improvement", msgs_per_sec / 3_000_000.0);

    // Display backtest metrics
    let metrics = &report.performance_metrics;
    println!("\n=== BACKTEST METRICS ===");
    println!("Total Trades: {}", metrics.total_trades);
    println!("Winning Trades: {} ({:.1}%)", 
        metrics.winning_trades, 
        if metrics.total_trades > 0 {
            (metrics.winning_trades as f64 / metrics.total_trades as f64) * 100.0
        } else {
            0.0
        }
    );
    println!("Total PnL: ${:.2}", metrics.total_pnl);
    println!("Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
    println!("Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    println!("Profit Factor: {:.2}", metrics.profit_factor);
    
    // Performance breakdown
    println!("\n=== PERFORMANCE BREAKDOWN ===");
    if msgs_per_sec < 3_000_000.0 {
        println!("⚠️  WARNING: Performance below 3M msgs/s target!");
        println!("   This is {}x slower than order book replay", 18_000_000.0 / msgs_per_sec);
        println!("   Bottlenecks likely in:");
        println!("   - Strategy calculations");
        println!("   - Position tracking");
        println!("   - Order management");
        println!("   - Feature extraction");
    } else {
        println!("✓ Performance exceeds 3M msgs/s target!");
        println!("  Achieved {:.1}x the baseline performance", msgs_per_sec / 3_000_000.0);
    }
}