//! Profile Serial Backtest Performance
//! 
//! This tool profiles where time is spent in serial backtesting
//! to identify optimization opportunities.

use algotrading::core::{MarketDataSource, MarketUpdate};
use algotrading::features::{FeaturePosition, RiskLimits};
use algotrading::market_data::FileReader;
use algotrading::market_data::events::{MarketEvent, TradeEvent};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use algotrading::strategy::{Strategy, StrategyContext};
use std::path::Path;
use std::time::{Instant, Duration};

#[derive(Default)]
struct ProfileStats {
    total_events: usize,
    file_reading: Duration,
    event_conversion: Duration,
    strategy_execution: Duration,
    order_processing: Duration,
    total_time: Duration,
}

fn main() {
    println!("=== BACKTEST PERFORMANCE PROFILING ===\n");

    let test_file = Path::new("../Market_Data/GLBX-20250528-84NHYCGUFY/glbx-mdp3-20250428.mbo.dbn.zst");
    if !test_file.exists() {
        eprintln!("Error: Test file not found");
        return;
    }

    // Test different optimization levels
    println!("1. Baseline Performance:");
    let baseline = profile_baseline(test_file);
    print_stats(&baseline);

    println!("\n2. Zero-Allocation Hot Path:");
    let zero_alloc = profile_zero_allocation(test_file);
    print_stats(&zero_alloc);

    println!("\n3. Batched Processing:");
    let batched = profile_batched(test_file);
    print_stats(&batched);

    println!("\n4. Inlined Strategy:");
    let inlined = profile_inlined_strategy(test_file);
    print_stats(&inlined);

    // Analysis
    println!("\n=== PERFORMANCE BREAKDOWN ===");
    let total_ns = baseline.total_time.as_nanos() as f64;
    println!("File Reading: {:.1}%", (baseline.file_reading.as_nanos() as f64 / total_ns) * 100.0);
    println!("Event Conversion: {:.1}%", (baseline.event_conversion.as_nanos() as f64 / total_ns) * 100.0);
    println!("Strategy Execution: {:.1}%", (baseline.strategy_execution.as_nanos() as f64 / total_ns) * 100.0);
    println!("Order Processing: {:.1}%", (baseline.order_processing.as_nanos() as f64 / total_ns) * 100.0);

    // Hardware limits
    println!("\n=== HARDWARE LIMIT ANALYSIS ===");
    let ns_per_event = baseline.total_time.as_nanos() / baseline.total_events as u128;
    println!("Nanoseconds per event: {}", ns_per_event);
    
    // Assuming 3GHz CPU
    let cycles_per_event = (ns_per_event as f64 * 3.0) as u64;
    println!("CPU cycles per event (3GHz): ~{}", cycles_per_event);
    
    if cycles_per_event < 1000 {
        println!("Status: Approaching hardware limits (< 1000 cycles/event)");
    } else if cycles_per_event < 5000 {
        println!("Status: Well optimized, limited headroom");
    } else {
        println!("Status: Significant optimization potential remains");
    }

    // Memory bandwidth estimate
    let bytes_per_event = 200; // Rough estimate
    let bandwidth_gb_s = (baseline.total_events * bytes_per_event) as f64 
        / baseline.total_time.as_secs_f64() / 1_000_000_000.0;
    println!("\nEstimated memory bandwidth: {:.1} GB/s", bandwidth_gb_s);
    println!("Typical DDR4 bandwidth: 25-50 GB/s");
    
    if bandwidth_gb_s > 20.0 {
        println!("Status: May be memory bandwidth limited");
    } else {
        println!("Status: Not memory bandwidth limited");
    }
}

fn profile_baseline(file: &Path) -> ProfileStats {
    let mut stats = ProfileStats::default();
    let total_start = Instant::now();

    // Initialize strategy
    let mut strategy = MeanReversionStrategy::new(
        "Baseline".to_string(),
        5921,
        MeanReversionConfig::default(),
    );
    
    let context = StrategyContext::new(
        "baseline".to_string(),
        0,
        FeaturePosition::default(),
        RiskLimits::default(),
        true,
    );
    
    strategy.initialize(&context).unwrap();

    // Process file
    let file_start = Instant::now();
    let mut reader = FileReader::new(vec![file.to_path_buf()]).unwrap();
    stats.file_reading = file_start.elapsed();

    while let Some(update) = reader.next_update() {
        let read_time = Instant::now();
        stats.file_reading += read_time.elapsed();

        // Convert event
        let convert_start = Instant::now();
        let event = match update {
            MarketUpdate::Trade(trade) => MarketEvent::Trade(TradeEvent {
                instrument_id: trade.instrument_id,
                trade_id: 0,
                price: trade.price,
                quantity: trade.quantity,
                aggressor_side: trade.side,
                timestamp: trade.timestamp,
                buyer_order_id: None,
                seller_order_id: None,
            }),
            _ => continue,
        };
        stats.event_conversion += convert_start.elapsed();

        // Execute strategy
        let strategy_start = Instant::now();
        let output = strategy.on_market_event(&event, &context);
        stats.strategy_execution += strategy_start.elapsed();

        // Process orders
        let order_start = Instant::now();
        for _order in &output.orders {
            // Simulate order processing
            std::hint::black_box(&_order);
        }
        stats.order_processing += order_start.elapsed();

        stats.total_events += 1;
    }

    stats.total_time = total_start.elapsed();
    stats
}

fn profile_zero_allocation(file: &Path) -> ProfileStats {
    let mut stats = ProfileStats::default();
    let total_start = Instant::now();

    // Pre-allocate everything
    let mut strategy = MeanReversionStrategy::new(
        "ZeroAlloc".to_string(),
        5921,
        MeanReversionConfig::default(),
    );
    
    let mut context = StrategyContext::new(
        "zero_alloc".to_string(),
        0,
        FeaturePosition::default(),
        RiskLimits::default(),
        true,
    );
    
    strategy.initialize(&context).unwrap();

    // Reusable event
    let mut event = MarketEvent::Trade(TradeEvent {
        instrument_id: 5921,
        trade_id: 0,
        price: algotrading::core::types::Price::new(0),
        quantity: algotrading::core::types::Quantity::from(1u32),
        aggressor_side: algotrading::core::Side::Bid,
        timestamp: 0,
        buyer_order_id: None,
        seller_order_id: None,
    });

    let mut reader = FileReader::new(vec![file.to_path_buf()]).unwrap();

    while let Some(update) = reader.next_update() {
        // Update existing event instead of creating new
        match update {
            MarketUpdate::Trade(trade) => {
                if let MarketEvent::Trade(ref mut t) = event {
                    t.price = trade.price;
                    t.quantity = trade.quantity;
                    t.aggressor_side = trade.side;
                    t.timestamp = trade.timestamp;
                }
                
                context.current_time = trade.timestamp;
                let _output = strategy.on_market_event(&event, &context);
                stats.total_events += 1;
            }
            _ => continue,
        }
    }

    stats.total_time = total_start.elapsed();
    stats
}

fn profile_batched(file: &Path) -> ProfileStats {
    let mut stats = ProfileStats::default();
    let total_start = Instant::now();
    
    const BATCH_SIZE: usize = 1000;

    let mut strategy = MeanReversionStrategy::new(
        "Batched".to_string(),
        5921,
        MeanReversionConfig::default(),
    );
    
    let context = StrategyContext::new(
        "batched".to_string(),
        0,
        FeaturePosition::default(),
        RiskLimits::default(),
        true,
    );
    
    strategy.initialize(&context).unwrap();

    let mut reader = FileReader::new(vec![file.to_path_buf()]).unwrap();
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    loop {
        // Fill batch
        batch.clear();
        for _ in 0..BATCH_SIZE {
            if let Some(update) = reader.next_update() {
                match update {
                    MarketUpdate::Trade(trade) => batch.push(trade),
                    MarketUpdate::OrderBook(_) => continue, // Skip for now
                }
            } else {
                break;
            }
        }

        if batch.is_empty() {
            break;
        }

        // Process batch - better cache locality
        for trade in &batch {
            let event = MarketEvent::Trade(TradeEvent {
                instrument_id: trade.instrument_id,
                trade_id: 0,
                price: trade.price,
                quantity: trade.quantity,
                aggressor_side: trade.side,
                timestamp: trade.timestamp,
                buyer_order_id: None,
                seller_order_id: None,
            });

            let _output = strategy.on_market_event(&event, &context);
            stats.total_events += 1;
        }
    }

    stats.total_time = total_start.elapsed();
    stats
}

fn profile_inlined_strategy(file: &Path) -> ProfileStats {
    let mut stats = ProfileStats::default();
    let total_start = Instant::now();

    // Inline the strategy logic instead of virtual dispatch
    let mut prices = Vec::with_capacity(20);
    let lookback = 20;
    let entry_threshold = 2.0;
    let mut position = 0i64;

    let mut reader = FileReader::new(vec![file.to_path_buf()]).unwrap();

    while let Some(update) = reader.next_update() {
        if let MarketUpdate::Trade(trade) = update {
            // Inlined mean reversion logic
            prices.push(trade.price.as_f64());
            if prices.len() > lookback {
                prices.remove(0);
            }

            if prices.len() == lookback {
                let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
                let variance: f64 = prices.iter()
                    .map(|p| (p - mean).powi(2))
                    .sum::<f64>() / prices.len() as f64;
                let std_dev = variance.sqrt();

                let current = trade.price.as_f64();
                let z_score = (current - mean) / std_dev;

                // Generate orders
                if z_score < -entry_threshold && position <= 0 {
                    position += 1; // Buy
                } else if z_score > entry_threshold && position >= 0 {
                    position -= 1; // Sell
                }
            }

            stats.total_events += 1;
        }
    }

    stats.total_time = total_start.elapsed();
    stats
}

fn print_stats(stats: &ProfileStats) {
    println!("  Events: {}", stats.total_events);
    println!("  Time: {:.3}s", stats.total_time.as_secs_f64());
    
    if stats.total_events > 0 {
        let throughput = stats.total_events as f64 / stats.total_time.as_secs_f64();
        println!("  Throughput: {:.2}M events/s", throughput / 1_000_000.0);
        println!("  Latency: {} ns/event", stats.total_time.as_nanos() / stats.total_events as u128);
    } else {
        println!("  Throughput: N/A (no events)");
        println!("  Latency: N/A");
    }
}