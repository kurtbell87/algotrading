//! Profile ALL Events (not just trades)
//! 
//! This gives us the true backtesting performance including order book updates

use algotrading::backtest::{BacktestConfig, BacktestEngine};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("=== TRUE BACKTEST PERFORMANCE (ALL EVENTS) ===\n");

    let test_file = Path::new("../Market_Data/GLBX-20250528-84NHYCGUFY/glbx-mdp3-20250428.mbo.dbn.zst");
    
    // Use the actual backtest engine to get true performance
    println!("Running full backtest with ALL market events...");
    let start = Instant::now();
    
    let config = BacktestConfig::default();
    let mut engine = BacktestEngine::new(config);
    
    let strategy = MeanReversionStrategy::new(
        "FullTest".to_string(),
        5921,
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
    
    engine.add_strategy(Box::new(strategy)).unwrap();
    
    let report = engine.run(&[test_file.to_path_buf()]).unwrap();
    let duration = start.elapsed();
    
    println!("\n=== RESULTS ===");
    println!("Total Events: {} (includes order book updates)", report.events_processed);
    println!("Total Time: {:.3}s", duration.as_secs_f64());
    println!("Throughput: {:.2}M events/s", report.events_processed as f64 / duration.as_secs_f64() / 1_000_000.0);
    println!("Nanoseconds per event: {}", duration.as_nanos() / report.events_processed as u128);
    
    // CPU cycle analysis
    let ns_per_event = duration.as_nanos() / report.events_processed as u128;
    let cycles_at_3ghz = (ns_per_event as f64 * 3.0) as u64;
    let cycles_at_5ghz = (ns_per_event as f64 * 5.0) as u64;
    
    println!("\n=== HARDWARE ANALYSIS ===");
    println!("CPU cycles per event:");
    println!("  @ 3 GHz: {} cycles", cycles_at_3ghz);
    println!("  @ 5 GHz: {} cycles", cycles_at_5ghz);
    
    if cycles_at_3ghz < 1000 {
        println!("\n🚨 AT HARDWARE LIMIT!");
        println!("   < 1000 cycles per event indicates we're CPU bound");
        println!("   Further optimization requires:");
        println!("   - Faster CPU (higher clock speed)");
        println!("   - Better CPU architecture (IPC improvements)");
        println!("   - Fundamental algorithm changes");
    } else if cycles_at_3ghz < 3000 {
        println!("\n⚡ HIGHLY OPTIMIZED");
        println!("   1000-3000 cycles per event");
        println!("   Limited optimization headroom remains");
        println!("   Consider:");
        println!("   - Profile-guided optimization (PGO)");
        println!("   - Hand-tuned assembly for hot paths");
        println!("   - Custom memory allocators");
    } else {
        println!("\n✅ OPTIMIZATION POSSIBLE");
        println!("   > 3000 cycles per event");
        println!("   Room for improvement exists");
    }
    
    // Memory bandwidth estimate
    let bytes_per_event = 256; // MBO message + strategy state
    let bandwidth_gb_s = (report.events_processed * bytes_per_event) as f64 
        / duration.as_secs_f64() / 1_000_000_000.0;
    
    println!("\n=== MEMORY BANDWIDTH ===");
    println!("Estimated usage: {:.1} GB/s", bandwidth_gb_s);
    println!("DDR4-3200 max: ~50 GB/s");
    println!("DDR5-5600 max: ~90 GB/s");
    
    if bandwidth_gb_s > 40.0 {
        println!("Status: May be memory bandwidth limited");
    } else {
        println!("Status: Not memory bandwidth limited");
    }
    
    // Theoretical limits
    println!("\n=== THEORETICAL LIMITS ===");
    println!("Current: {:.1}M events/s", report.events_processed as f64 / duration.as_secs_f64() / 1_000_000.0);
    println!("If 500 cycles/event: {:.1}M events/s @ 3GHz", 3_000_000_000.0 / 500.0 / 1_000_000.0);
    println!("If 500 cycles/event: {:.1}M events/s @ 5GHz", 5_000_000_000.0 / 500.0 / 1_000_000.0);
    println!("If 100 cycles/event: {:.1}M events/s @ 5GHz", 5_000_000_000.0 / 100.0 / 1_000_000.0);
}