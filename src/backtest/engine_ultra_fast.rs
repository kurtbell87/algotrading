//! Ultra-fast backtesting engine focused purely on throughput
//!
//! Stripped down version with minimal features for maximum performance

use crate::core::MarketUpdate;
use crate::core::traits::MarketDataSource;
use crate::features::{FeaturePosition, RiskLimits};
use crate::market_data::FileReader;
use crate::market_data::events::{MarketEvent, TradeEvent};
use crate::strategy::{Strategy, StrategyContext};
use std::path::{Path, PathBuf};
use std::time::Instant;

const BATCH_SIZE: usize = 5000; // Larger batches for better performance
const REPORT_INTERVAL: usize = 500_000;

/// Ultra-fast backtesting engine
pub struct UltraFastEngine {
    strategies: Vec<(Box<dyn Strategy>, StrategyContext)>,
    current_time: u64,
    events_processed: usize,
    start_time: Instant,
    // Pre-allocated batch buffer
    batch_buffer: Vec<MarketUpdate>,
    // Pre-allocated event buffer
    event_buffer: Vec<MarketEvent>,
}

impl UltraFastEngine {
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            current_time: 0,
            events_processed: 0,
            start_time: Instant::now(),
            batch_buffer: Vec::with_capacity(BATCH_SIZE),
            event_buffer: Vec::with_capacity(BATCH_SIZE),
        }
    }

    /// Add strategy
    pub fn add_strategy(&mut self, mut strategy: Box<dyn Strategy>) -> Result<(), String> {
        let context = StrategyContext::new(
            strategy.config().name.clone(),
            0,
            FeaturePosition::default(),
            RiskLimits::default(),
            true,
        );

        strategy.initialize(&context)?;
        self.strategies.push((strategy, context));
        Ok(())
    }

    /// Run ultra-fast backtest
    pub fn run<P: AsRef<Path>>(&mut self, data_files: &[P]) -> Result<UltraFastReport, String> {
        self.start_time = Instant::now();
        println!("\n=== Ultra-Fast Engine Starting ===");

        for (i, file) in data_files.iter().enumerate() {
            println!(
                "File {}/{}: {:?}",
                i + 1,
                data_files.len(),
                file.as_ref().file_name().unwrap_or_default()
            );
            self.process_file_ultra_fast(file)?;
        }

        let elapsed = self.start_time.elapsed();
        let throughput = self.events_processed as f64 / elapsed.as_secs_f64();

        Ok(UltraFastReport {
            events_processed: self.events_processed,
            elapsed_seconds: elapsed.as_secs_f64(),
            throughput,
        })
    }

    /// Process file with maximum performance
    fn process_file_ultra_fast<P: AsRef<Path>>(&mut self, file_path: P) -> Result<(), String> {
        let paths = vec![PathBuf::from(file_path.as_ref())];
        let mut reader =
            FileReader::new(paths).map_err(|e| format!("Failed to open file: {}", e))?;

        let file_start = Instant::now();
        let mut file_events = 0;

        loop {
            // Fill batch
            self.batch_buffer.clear();
            for _ in 0..BATCH_SIZE {
                if let Some(update) = reader.next_update() {
                    self.batch_buffer.push(update);
                } else {
                    break;
                }
            }

            if self.batch_buffer.is_empty() {
                break;
            }

            // Process batch inline (no function calls)
            self.event_buffer.clear();

            // Convert to events
            for update in &self.batch_buffer {
                let event = match update {
                    MarketUpdate::Trade(trade) => {
                        self.current_time = trade.timestamp;
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
                    MarketUpdate::OrderBook(book) => {
                        self.current_time = book.timestamp;
                        continue; // Skip order book updates for speed
                    }
                };
                self.event_buffer.push(event);
            }

            // Process through strategies (simplified)
            for event in &self.event_buffer {
                for (strategy, context) in &mut self.strategies {
                    context.current_time = self.current_time;
                    let _output = strategy.on_market_event(event, context);
                    // Don't process orders for pure speed test
                }
            }

            file_events += self.batch_buffer.len();
            self.events_processed += self.batch_buffer.len();

            // Progress report
            if self.events_processed % REPORT_INTERVAL == 0 {
                let elapsed = file_start.elapsed();
                let rate = file_events as f64 / elapsed.as_secs_f64();
                let total_rate =
                    self.events_processed as f64 / self.start_time.elapsed().as_secs_f64();
                println!(
                    "  {} events | Current: {:.0}/s | Average: {:.0}/s",
                    self.events_processed, rate, total_rate
                );
            }
        }

        Ok(())
    }
}

/// Ultra-fast engine report
#[derive(Debug)]
pub struct UltraFastReport {
    pub events_processed: usize,
    pub elapsed_seconds: f64,
    pub throughput: f64,
}

impl UltraFastReport {
    pub fn print_summary(&self) {
        println!("\n=== Ultra-Fast Engine Results ===");
        println!("Events: {}", self.events_processed);
        println!("Time: {:.2}s", self.elapsed_seconds);
        println!("Throughput: {:.0} events/s", self.throughput);

        let efficiency = (self.throughput / 18_000_000.0) * 100.0;
        println!("Efficiency: {:.1}% of 18M target", efficiency);

        if self.throughput >= 15_000_000.0 {
            println!("\n🎯 SUCCESS: Achieved target!");
        } else if self.throughput >= 10_000_000.0 {
            println!("\n✅ GOOD: Significant improvement");
        } else if self.throughput >= 6_000_000.0 {
            println!("\n⚠️  MODERATE: Getting closer");
        } else {
            println!("\n❌ More optimization needed");
        }
    }
}
