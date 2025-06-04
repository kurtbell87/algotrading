//! Shared utilities for trading strategies
//!
//! This module contains common calculations and constants used across multiple strategies.

use crate::core::types::{InstrumentId, Price};
use std::collections::HashMap;

/// Default tick size for instruments (in scaled units)
/// TODO: This should be moved to instrument configuration
/// This is 0.025 dollars = 25_000_000 in fixed-point (1e9 scale)
pub const DEFAULT_TICK_SIZE: i64 = 25_000_000;

/// Get tick size for an instrument
/// Currently returns default tick size, but should be replaced with instrument-specific configuration
#[inline]
pub fn get_tick_size(_instrument_id: InstrumentId) -> i64 {
    DEFAULT_TICK_SIZE
}

/// Round a price to the nearest tick
#[inline]
pub fn round_to_tick(price: Price, tick_size: i64) -> Price {
    let ticks = (price.0 + tick_size / 2) / tick_size;
    Price::new(ticks * tick_size)
}

/// Calculate mid price from bid and ask
#[inline]
pub fn calculate_mid_price(bid: Price, ask: Price) -> Price {
    Price::from_f64((bid.as_f64() + ask.as_f64()) / 2.0)
}

/// Calculate spread in ticks
#[inline]
pub fn calculate_spread_ticks(bid: Price, ask: Price, tick_size: i64) -> i64 {
    (ask.0 - bid.0) / tick_size
}

/// Calculate simple moving average
#[inline]
pub fn calculate_sma(prices: &[(u64, Price)], period: usize) -> Option<f64> {
    if prices.len() < period {
        return None;
    }
    
    let sum: f64 = prices
        .iter()
        .rev()
        .take(period)
        .map(|(_, p)| p.as_f64())
        .sum();
    
    Some(sum / period as f64)
}

/// Calculate exponential moving average
#[inline]
pub fn calculate_ema(prices: &[(u64, Price)], period: usize, prev_ema: Option<f64>) -> Option<f64> {
    if prices.is_empty() {
        return None;
    }
    
    let current_price = prices.last().unwrap().1.as_f64();
    
    match prev_ema {
        Some(prev) => {
            let alpha = 2.0 / (period as f64 + 1.0);
            Some(alpha * current_price + (1.0 - alpha) * prev)
        }
        None => {
            // Initialize with SMA
            calculate_sma(prices, period)
        }
    }
}

/// Calculate standard deviation
#[inline]
pub fn calculate_std_dev(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let variance = values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    
    variance.sqrt()
}

/// Position sizing based on Kelly criterion (simplified)
#[inline]
pub fn kelly_position_size(
    win_rate: f64,
    avg_win: f64,
    avg_loss: f64,
    max_position: i64,
    scale_factor: f64,
) -> i64 {
    if avg_loss <= 0.0 || win_rate <= 0.0 || win_rate >= 1.0 {
        return 0;
    }
    
    let loss_rate = 1.0 - win_rate;
    let win_loss_ratio = avg_win / avg_loss;
    
    // Kelly formula: f = (p * b - q) / b
    // where p = win_rate, q = loss_rate, b = win_loss_ratio
    let kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio;
    
    // Apply scale factor (typically 0.25 for 1/4 Kelly)
    let position_fraction = kelly_fraction * scale_factor;
    
    // Clamp to reasonable bounds
    let position_size = (max_position as f64 * position_fraction).round() as i64;
    position_size.clamp(0, max_position)
}

/// Instrument configuration (placeholder for future implementation)
#[derive(Debug, Clone)]
pub struct InstrumentConfig {
    pub id: InstrumentId,
    pub tick_size: i64,
    pub min_price_increment: i64,
    pub lot_size: u32,
    pub max_position: i64,
}

impl Default for InstrumentConfig {
    fn default() -> Self {
        Self {
            id: 0,
            tick_size: DEFAULT_TICK_SIZE,
            min_price_increment: DEFAULT_TICK_SIZE,
            lot_size: 1,
            max_position: 100,
        }
    }
}

/// Instrument configuration manager (placeholder)
pub struct InstrumentConfigManager {
    configs: HashMap<InstrumentId, InstrumentConfig>,
}

impl InstrumentConfigManager {
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }
    
    pub fn get(&self, instrument_id: InstrumentId) -> &InstrumentConfig {
        static DEFAULT_CONFIG: InstrumentConfig = InstrumentConfig {
            id: 0,
            tick_size: DEFAULT_TICK_SIZE,
            min_price_increment: DEFAULT_TICK_SIZE,
            lot_size: 1,
            max_position: 100,
        };
        
        self.configs.get(&instrument_id).unwrap_or(&DEFAULT_CONFIG)
    }
    
    pub fn add(&mut self, config: InstrumentConfig) {
        self.configs.insert(config.id, config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_round_to_tick() {
        // Price::new expects raw fixed-point values (scaled by 1e9)
        // So 100.37 dollars = 100_370_000_000 in fixed point
        // tick_size of 25 = 0.000000025 in fixed point
        let tick_size = 25_000_000; // 0.025 dollars in fixed point
        
        let price = Price::new(100_370_000_000); // 100.37
        let rounded = round_to_tick(price, tick_size);
        assert_eq!(rounded.0, 100_375_000_000); // Rounds to 100.375
        
        let price2 = Price::new(100_012_000_000); // 100.012
        let rounded2 = round_to_tick(price2, tick_size);
        assert_eq!(rounded2.0, 100_000_000_000); // Rounds to 100.000
    }
    
    #[test]
    fn test_calculate_mid_price() {
        let bid = Price::new(10000);
        let ask = Price::new(10050);
        let mid = calculate_mid_price(bid, ask);
        assert_eq!(mid.0, 10025);
    }
    
    #[test]
    fn test_calculate_spread_ticks() {
        let bid = Price::new(10000);
        let ask = Price::new(10050);
        let spread = calculate_spread_ticks(bid, ask, 25);
        assert_eq!(spread, 2); // 50 / 25 = 2 ticks
    }
    
    #[test]
    fn test_calculate_sma() {
        let prices = vec![
            (1000, Price::from(100i64)),
            (2000, Price::from(102i64)),
            (3000, Price::from(104i64)),
            (4000, Price::from(103i64)),
            (5000, Price::from(105i64)),
        ];
        
        let sma = calculate_sma(&prices, 3);
        assert!(sma.is_some());
        // Should be average of last 3: (104 + 103 + 105) / 3 = 104
        assert!((sma.unwrap() - 104.0).abs() < 0.1);
    }
    
    #[test]
    fn test_kelly_position_size() {
        // 60% win rate, 2:1 win/loss ratio
        let size = kelly_position_size(0.6, 2.0, 1.0, 100, 0.25);
        // Kelly = (0.6 * 2 - 0.4) / 2 = 0.8 / 2 = 0.4
        // Quarter Kelly = 0.4 * 0.25 = 0.1
        // Position = 100 * 0.1 = 10
        assert_eq!(size, 10);
        
        // Edge case: no edge
        let size = kelly_position_size(0.5, 1.0, 1.0, 100, 0.25);
        assert_eq!(size, 0);
    }
}