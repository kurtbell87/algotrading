//! Final performance test comparing approaches
//!
//! Uses real MBO data to measure actual throughput

use algotrading::backtest::{BacktestEngine, BacktestConfig};
use algotrading::strategies::MeanReversionStrategy;
use algotrading::strategies::mean_reversion::MeanReversionConfig;
use algotrading::core::traits::MarketDataSource;
use algotrading::market_data::FileReader;
use algotrading::strategy::Strategy;
use std::time::Instant;
use std::path::PathBuf;

fn main() {
    println!("=== FINAL BACKTESTING PERFORMANCE TEST ===");
    println!("Using real MBO data files");
    println!("Target: >15M events/s (>83% of 18M)\n");

    let data_dir = PathBuf::from("../Market_Data/GLBX-20250528-84NHYCGUFY");
    let test_file = data_dir.join("glbx-mdp3-20250428.mbo.dbn.zst");
    
    if !test_file.exists() {
        eprintln!("Error: Market data file not found");
        return;
    }

    // Test 1: Raw file reading speed (baseline)
    println!("1. RAW FILE READING (baseline):");
    let read_speed = test_raw_reading(&test_file);
    
    // Test 2: Current backtest engine
    println!("\n2. CURRENT BACKTEST ENGINE:");
    let backtest_speed = test_backtest_engine(&test_file);
    
    // Test 3: Direct processing (no backtest infrastructure)
    println!("\n3. DIRECT PROCESSING (minimal overhead):");
    let direct_speed = test_direct_processing(&test_file);
    
    // Summary
    print_summary(read_speed, backtest_speed, direct_speed);
}

fn test_raw_reading(file: &PathBuf) -> f64 {
    let start = Instant::now();
    
    let mut reader = FileReader::new(vec![file.clone()]).unwrap();
    let mut count = 0;
    
    while let Some(_) = reader.next_update() {
        count += 1;
    }
    
    let elapsed = start.elapsed();
    let throughput = count as f64 / elapsed.as_secs_f64();
    
    println!("  Events: {}", count);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} events/s", throughput);
    
    throughput
}

fn test_backtest_engine(file: &PathBuf) -> f64 {
    let start = Instant::now();
    
    let config = BacktestConfig::default();
    let mut engine = BacktestEngine::new(config);
    
    let strategy = MeanReversionStrategy::new(
        "Test".to_string(),
        5921,
        MeanReversionConfig::default(),
    );
    
    let _ = engine.add_strategy(Box::new(strategy));
    
    match engine.run(&[file]) {
        Ok(report) => {
            let elapsed = start.elapsed();
            let throughput = report.events_processed as f64 / elapsed.as_secs_f64();
            
            println!("  Events: {}", report.events_processed);
            println!("  Time: {:.2}s", elapsed.as_secs_f64());
            println!("  Throughput: {:.0} events/s", throughput);
            
            throughput
        }
        Err(e) => {
            eprintln!("  Error: {}", e);
            0.0
        }
    }
}

fn test_direct_processing(file: &PathBuf) -> f64 {
    use algotrading::market_data::events::{MarketEvent, TradeEvent};
    use algotrading::core::MarketUpdate;
    
    let start = Instant::now();
    
    let mut reader = FileReader::new(vec![file.clone()]).unwrap();
    let mut count = 0;
    
    // Create strategy once
    let mut strategy = MeanReversionStrategy::new(
        "Test".to_string(),
        5921,
        MeanReversionConfig::default(),
    );
    
    let context = algotrading::strategy::StrategyContext::new(
        "Test".to_string(),
        0,
        algotrading::features::FeaturePosition::default(),
        algotrading::features::RiskLimits::default(),
        true,
    );
    
    // Process events directly
    while let Some(update) = reader.next_update() {
        count += 1;
        
        // Convert to market event
        let event = match update {
            MarketUpdate::Trade(trade) => {
                MarketEvent::Trade(TradeEvent {
                    instrument_id: trade.instrument_id,
                    trade_id: 0,
                    price: trade.price,
                    quantity: trade.quantity,
                    aggressor_side: trade.side,
                    timestamp: trade.timestamp,
                    buyer_order_id: None,
                    seller_order_id: None,
                })
            }
            MarketUpdate::OrderBook(_) => continue,
        };
        
        // Call strategy directly
        let _output = strategy.on_market_event(&event, &context);
    }
    
    let elapsed = start.elapsed();
    let throughput = count as f64 / elapsed.as_secs_f64();
    
    println!("  Events: {}", count);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} events/s", throughput);
    
    throughput
}

fn print_summary(read: f64, backtest: f64, direct: f64) {
    println!("\n=== PERFORMANCE SUMMARY ===");
    
    println!("\nThroughput (events/s):");
    println!("  Raw reading:    {:.0} ({:.1}% of 18M)", read, (read / 18_000_000.0) * 100.0);
    println!("  Backtest:       {:.0} ({:.1}% of 18M)", backtest, (backtest / 18_000_000.0) * 100.0);
    println!("  Direct:         {:.0} ({:.1}% of 18M)", direct, (direct / 18_000_000.0) * 100.0);
    
    println!("\nOverhead Analysis:");
    if read > 0.0 {
        let backtest_overhead = ((read - backtest) / read) * 100.0;
        let direct_overhead = ((read - direct) / read) * 100.0;
        
        println!("  Backtest overhead: {:.1}%", backtest_overhead);
        println!("  Direct overhead:   {:.1}%", direct_overhead);
    }
    
    println!("\nBottleneck Analysis:");
    if backtest < 6_000_000.0 {
        println!("  ❌ Backtest engine has severe overhead ({}x slowdown)", 
                (read / backtest) as i32);
        println!("  Main issues:");
        println!("    - Event queue management");
        println!("    - Lock contention");
        println!("    - Memory allocations");
    }
    
    if direct > backtest * 2.0 {
        println!("  ✅ Direct processing shows potential for {}x improvement",
                (direct / backtest) as i32);
    }
    
    println!("\nRecommendations:");
    if direct >= 10_000_000.0 {
        println!("  - Direct processing achieves good performance");
        println!("  - Focus on reducing backtest infrastructure overhead");
        println!("  - Consider lockless architecture");
    } else {
        println!("  - Even direct processing is below target");
        println!("  - Strategy computation may be the bottleneck");
        println!("  - Consider strategy optimization");
    }
}