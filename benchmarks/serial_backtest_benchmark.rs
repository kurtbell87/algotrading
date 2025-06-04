//! Serial Backtest Benchmark with Pre-computed Features
//! 
//! This benchmark shows the CORRECT way to optimize backtesting:
//! 1. Pre-compute features in parallel (can be saved/loaded)
//! 2. Run strategy sequentially with proper state management

use algotrading::core::{MarketDataSource, MarketUpdate};
use algotrading::features::{FeaturePosition, RiskLimits};
use algotrading::market_data::FileReader;
use algotrading::market_data::events::{MarketEvent, TradeEvent};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use algotrading::strategy::{Strategy, StrategyContext, OrderSide};
use algotrading::core::types::Price;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
struct PrecomputedFeatures {
    timestamp: u64,
    bid_price: Option<Price>,
    ask_price: Option<Price>,
    mid_price: Price,
    spread: i64,
    volume_imbalance: f64,
    // Add more features as needed
}

fn main() {
    println!("=== SERIAL BACKTEST BENCHMARK ===");
    println!("Demonstrating the CORRECT way to optimize backtesting\n");

    // Get the market data directory
    let data_dir = Path::new("../Market_Data/GLBX-20250528-84NHYCGUFY");
    
    // Get all .mbo.dbn.zst files
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("zst") &&
               path.to_string_lossy().contains(".mbo.dbn") {
                files.push(path);
            }
        }
    }
    files.sort();
    
    if files.is_empty() {
        eprintln!("Error: No MBO files found in {:?}", data_dir);
        return;
    }

    println!("Found {} MBO files", files.len());
    
    // Step 1: Pre-compute features (this CAN be parallelized)
    println!("\n1. Pre-computing features (parallelizable)...");
    let features_start = Instant::now();
    let all_features = precompute_features_parallel(&files);
    let features_duration = features_start.elapsed();
    println!("  Computed {} feature records in {:.2}s", all_features.len(), features_duration.as_secs_f64());
    println!("  Throughput: {:.1}M records/s", all_features.len() as f64 / features_duration.as_secs_f64() / 1_000_000.0);
    
    // Step 2: Run SERIAL backtest
    println!("\n2. Running SERIAL backtest (must be sequential)...");
    let backtest_start = Instant::now();
    let results = run_serial_backtest(&all_features);
    let backtest_duration = backtest_start.elapsed();
    
    // Results
    println!("\n=== BACKTEST RESULTS ===");
    println!("Events Processed: {}", results.events_processed);
    println!("Backtest Time: {:.2}s", backtest_duration.as_secs_f64());
    println!("Throughput: {:.1}M events/s", results.events_processed as f64 / backtest_duration.as_secs_f64() / 1_000_000.0);
    println!("\nTrades Executed: {}", results.trades_executed);
    println!("Final Position: {}", results.final_position);
    println!("Total PnL: ${:.2}", results.total_pnl);
    println!("Sharpe Ratio: {:.2}", results.sharpe_ratio);
    println!("Max Drawdown: {:.2}%", results.max_drawdown * 100.0);
    
    // Performance comparison
    println!("\n=== PERFORMANCE ANALYSIS ===");
    let total_time = features_duration + backtest_duration;
    let overall_throughput = results.events_processed as f64 / total_time.as_secs_f64();
    println!("Total Time (features + backtest): {:.2}s", total_time.as_secs_f64());
    println!("Overall Throughput: {:.1}M events/s", overall_throughput / 1_000_000.0);
    println!("vs 3M baseline: {:.1}x improvement", overall_throughput / 3_000_000.0);
    
    // Multi-year estimate
    let events_per_day = 12_000_000u64;
    let trading_days_per_year = 252u64;
    let years = 5u64;
    let total_events = events_per_day * trading_days_per_year * years;
    let estimated_time = total_events as f64 / overall_throughput;
    
    println!("\n=== 5-YEAR BACKTEST ESTIMATE ===");
    println!("Total events: {:.1}B", total_events as f64 / 1_000_000_000.0);
    println!("Estimated time: {:.1} minutes", estimated_time / 60.0);
    println!("With Python/ML overhead: {:.1} minutes", estimated_time * 1.5 / 60.0);
    
    // Note about Arrow
    println!("\n=== OPTIMIZATION NOTES ===");
    println!("• Features can be pre-computed ONCE and saved to Arrow/Parquet");
    println!("• Subsequent backtests only need to load features (seconds)");
    println!("• Different strategies can reuse the same feature set");
    println!("• Only the serial backtest needs to be re-run for parameter changes");
}

fn precompute_features_parallel(files: &[PathBuf]) -> Vec<PrecomputedFeatures> {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let all_features = Arc::new(Mutex::new(Vec::new()));
    let num_threads = num_cpus::get();
    let chunks: Vec<_> = files.chunks(files.len() / num_threads + 1).collect();
    let mut handles = vec![];
    
    for chunk in chunks {
        let chunk = chunk.to_vec();
        let features = Arc::clone(&all_features);
        
        let handle = thread::spawn(move || {
            let mut local_features = Vec::new();
            
            for file in chunk {
                let mut reader = FileReader::new(vec![file]).unwrap();
                let mut last_bid = Price::new(0);
                let mut last_ask = Price::new(0);
                let mut volume_bid = 0u64;
                let mut volume_ask = 0u64;
                
                while let Some(update) = reader.next_update() {
                    match update {
                        MarketUpdate::Trade(trade) => {
                            // Update volume
                            match trade.side {
                                algotrading::core::Side::Bid => volume_bid += trade.quantity.0 as u64,
                                algotrading::core::Side::Ask => volume_ask += trade.quantity.0 as u64,
                            }
                            
                            // Compute features
                            let mid_price = if last_bid.0 > 0 && last_ask.0 > 0 {
                                Price::new((last_bid.0 + last_ask.0) / 2)
                            } else {
                                trade.price
                            };
                            
                            let spread = if last_bid.0 > 0 && last_ask.0 > 0 {
                                last_ask.0 - last_bid.0
                            } else {
                                0
                            };
                            
                            let volume_imbalance = if volume_bid + volume_ask > 0 {
                                (volume_bid as f64 - volume_ask as f64) / (volume_bid + volume_ask) as f64
                            } else {
                                0.0
                            };
                            
                            local_features.push(PrecomputedFeatures {
                                timestamp: trade.timestamp,
                                bid_price: if last_bid.0 > 0 { Some(last_bid) } else { None },
                                ask_price: if last_ask.0 > 0 { Some(last_ask) } else { None },
                                mid_price,
                                spread,
                                volume_imbalance,
                            });
                        }
                        MarketUpdate::OrderBook(_) => {
                            // In real implementation, update bid/ask from order book
                        }
                    }
                }
            }
            
            features.lock().unwrap().extend(local_features);
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let mut features = Arc::try_unwrap(all_features).unwrap().into_inner().unwrap();
    // Sort by timestamp (critical for serial processing!)
    features.sort_by_key(|f| f.timestamp);
    features
}

#[derive(Debug)]
struct BacktestResults {
    events_processed: usize,
    trades_executed: usize,
    final_position: i64,
    total_pnl: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
}

fn run_serial_backtest(features: &[PrecomputedFeatures]) -> BacktestResults {
    // Initialize strategy
    let mut strategy = MeanReversionStrategy::new(
        "SerialBacktest".to_string(),
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
    
    let mut context = StrategyContext::new(
        "serial".to_string(),
        0,
        FeaturePosition::default(),
        RiskLimits::default(),
        true,
    );
    
    strategy.initialize(&context).unwrap();
    
    // Backtest state (persists across entire period)
    let mut position = 0i64;
    let mut cash = 100_000.0; // Starting capital
    let mut trades_executed = 0;
    let mut pnl_history = Vec::new();
    let mut max_equity = cash;
    let mut max_drawdown = 0.0;
    
    // Process events SEQUENTIALLY
    for (i, feature) in features.iter().enumerate() {
        // Update context
        context.current_time = feature.timestamp;
        
        // Create market event from features
        let event = MarketEvent::Trade(TradeEvent {
            instrument_id: 5921,
            trade_id: i as u64,
            price: feature.mid_price,
            quantity: algotrading::core::types::Quantity::from(1u32),
            aggressor_side: algotrading::core::Side::Bid,
            timestamp: feature.timestamp,
            buyer_order_id: None,
            seller_order_id: None,
        });
        
        // Get strategy signal
        let output = strategy.on_market_event(&event, &context);
        
        // Execute orders (simplified - no slippage/commission for now)
        for order in &output.orders {
            let fill_price = feature.mid_price;
            let quantity = order.quantity.0 as i64;
            
            match order.side {
                OrderSide::Buy | OrderSide::BuyCover => {
                    position += quantity;
                    cash -= fill_price.as_f64() * quantity as f64;
                }
                OrderSide::Sell | OrderSide::SellShort => {
                    position -= quantity;
                    cash += fill_price.as_f64() * quantity as f64;
                }
            }
            trades_executed += 1;
        }
        
        // Calculate equity and track drawdown
        let unrealized_pnl = position as f64 * feature.mid_price.as_f64();
        let equity = cash + unrealized_pnl;
        pnl_history.push(equity - 100_000.0); // PnL relative to starting capital
        
        if equity > max_equity {
            max_equity = equity;
        }
        let drawdown = (max_equity - equity) / max_equity;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    
    // Calculate final metrics
    let total_pnl = pnl_history.last().copied().unwrap_or(0.0);
    let sharpe_ratio = calculate_sharpe_ratio(&pnl_history);
    
    BacktestResults {
        events_processed: features.len(),
        trades_executed,
        final_position: position,
        total_pnl,
        sharpe_ratio,
        max_drawdown,
    }
}

fn calculate_sharpe_ratio(pnl_history: &[f64]) -> f64 {
    if pnl_history.len() < 2 {
        return 0.0;
    }
    
    // Calculate returns
    let mut returns = Vec::new();
    for i in 1..pnl_history.len() {
        let return_pct = (pnl_history[i] - pnl_history[i-1]) / (100_000.0 + pnl_history[i-1]);
        returns.push(return_pct);
    }
    
    // Calculate mean and std dev
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();
    
    if std_dev > 0.0 {
        // Annualize (assuming minute bars, ~390 minutes per day, 252 days per year)
        let minutes_per_year = 390.0 * 252.0;
        let annualized_return = mean * minutes_per_year;
        let annualized_vol = std_dev * minutes_per_year.sqrt();
        annualized_return / annualized_vol
    } else {
        0.0
    }
}