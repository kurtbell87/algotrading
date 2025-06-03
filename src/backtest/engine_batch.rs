//! Batch processing optimization for backtesting engine
//!
//! This module provides batch processing utilities that can be used
//! with the existing BacktestEngine to improve performance.

use crate::core::MarketUpdate;
use crate::market_data::events::MarketEvent;
use crate::backtest::market_state::MarketStateManager;
use std::sync::{Arc, RwLock};

/// Batch processor for market updates
pub struct BatchProcessor {
    batch_size: usize,
    buffer: Vec<MarketUpdate>,
}

impl BatchProcessor {
    /// Create new batch processor
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            buffer: Vec::with_capacity(batch_size * 2),
        }
    }
    
    /// Add update to batch
    pub fn add(&mut self, update: MarketUpdate) {
        self.buffer.push(update);
    }
    
    /// Check if batch is full
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.batch_size
    }
    
    /// Process batch with callback
    pub fn process_batch<F>(&mut self, mut callback: F) 
    where
        F: FnMut(&[MarketUpdate])
    {
        if !self.buffer.is_empty() {
            callback(&self.buffer);
            self.buffer.clear();
        }
    }
    
    /// Get current batch size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

/// Optimized market state updater
pub struct BatchMarketUpdater {
    market_state: Arc<RwLock<MarketStateManager>>,
}

impl BatchMarketUpdater {
    pub fn new(market_state: Arc<RwLock<MarketStateManager>>) -> Self {
        Self { market_state }
    }
    
    /// Update market state for entire batch (single lock acquisition)
    pub fn update_batch(&self, updates: &[MarketUpdate]) {
        let mut state = self.market_state.write().unwrap();
        
        for update in updates {
            // Direct update without intermediate conversion
            match update {
                MarketUpdate::Trade(trade) => {
                    // Update last trade price
                    // In real implementation, update order book
                }
                MarketUpdate::OrderBook(book_update) => {
                    // Update order book levels
                }
            }
        }
    }
}

/// Performance measurement utilities
pub struct PerformanceMonitor {
    start_time: std::time::Instant,
    events_processed: usize,
    last_report_time: std::time::Instant,
    report_interval: usize,
}

impl PerformanceMonitor {
    pub fn new(report_interval: usize) -> Self {
        let now = std::time::Instant::now();
        Self {
            start_time: now,
            events_processed: 0,
            last_report_time: now,
            report_interval,
        }
    }
    
    /// Update event count
    pub fn add_events(&mut self, count: usize) {
        self.events_processed += count;
        
        if self.events_processed % self.report_interval == 0 {
            self.print_progress();
        }
    }
    
    /// Print progress report
    pub fn print_progress(&mut self) {
        let now = std::time::Instant::now();
        let interval_elapsed = now.duration_since(self.last_report_time);
        let total_elapsed = now.duration_since(self.start_time);
        
        let interval_rate = self.report_interval as f64 / interval_elapsed.as_secs_f64();
        let total_rate = self.events_processed as f64 / total_elapsed.as_secs_f64();
        
        println!("Processed {} events | Current: {:.0} events/s | Average: {:.0} events/s",
                self.events_processed, interval_rate, total_rate);
        
        self.last_report_time = now;
    }
    
    /// Get final statistics
    pub fn get_stats(&self) -> (usize, f64) {
        let elapsed = self.start_time.elapsed();
        let rate = self.events_processed as f64 / elapsed.as_secs_f64();
        (self.events_processed, rate)
    }
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of events per batch
    pub batch_size: usize,
    /// Enable progress reporting
    pub enable_progress: bool,
    /// Progress report interval
    pub report_interval: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            enable_progress: true,
            report_interval: 100_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_processor() {
        let mut processor = BatchProcessor::new(3);
        
        assert!(!processor.is_full());
        
        // Add updates
        for i in 0..3 {
            processor.add(MarketUpdate::Trade(crate::core::Trade {
                instrument_id: 1,
                price: crate::core::types::Price::from(100i64),
                quantity: crate::core::types::Quantity::from(1u32),
                side: crate::core::Side::Bid,
                timestamp: i as u64,
            }));
        }
        
        assert!(processor.is_full());
        assert_eq!(processor.len(), 3);
        
        // Process batch
        let mut processed = false;
        processor.process_batch(|batch| {
            assert_eq!(batch.len(), 3);
            processed = true;
        });
        
        assert!(processed);
        assert_eq!(processor.len(), 0);
    }
}