//! Event-based rolling features
//!
//! This module implements event-driven window calculations including:
//! - Event-based VWAP (Volume Weighted Average Price)
//! - Volume-based windows
//! - Rolling volatility over N events
//! - Exponentially weighted moving averages by event count
//!
//! Key principle: Markets are driven by events (orders/trades), not time.
//! All features update based on market events, not arbitrary time intervals.

use crate::core::types::{Price, Quantity};
use crate::features::collector::FeatureVector;
use std::collections::VecDeque;

/// Window type for event-based calculations
#[derive(Debug, Clone)]
pub enum WindowType {
    /// Last N events (trades or book updates)
    EventCount(usize),
    /// Last N contracts traded
    VolumeCount(u64),
    /// Last N trades
    TradeCount(usize),
    /// Exponentially weighted by event age
    ExponentialDecay { half_life_events: usize },
}

/// Price and volume data point
#[derive(Debug, Clone)]
struct PricePoint {
    #[allow(dead_code)]
    timestamp: u64,
    price: Price,
    volume: Option<Quantity>,
    event_number: u64,
}

/// Event-based rolling window
pub struct RollingWindow {
    /// Window type and size
    window_type: WindowType,
    /// Data points in window
    points: VecDeque<PricePoint>,
    /// Current event number
    event_counter: u64,
    /// Accumulated volume in window
    total_volume: f64,
    /// Trade count in window
    trade_count: usize,
    /// Cached VWAP value
    cached_vwap: Option<f64>,
    /// Cached volatility value
    cached_volatility: Option<f64>,
}

impl RollingWindow {
    pub fn new(window_type: WindowType) -> Self {
        Self {
            window_type,
            points: VecDeque::new(),
            event_counter: 0,
            total_volume: 0.0,
            trade_count: 0,
            cached_vwap: None,
            cached_volatility: None,
        }
    }

    /// Add a price point (called on every MBO event)
    pub fn add_point(&mut self, timestamp: u64, price: Price, volume: Option<Quantity>) {
        // Invalidate cache
        self.cached_vwap = None;
        self.cached_volatility = None;

        self.event_counter += 1;

        let point = PricePoint {
            timestamp,
            price,
            volume,
            event_number: self.event_counter,
        };

        // Update window statistics
        if let Some(vol) = volume {
            self.total_volume += vol.as_f64();
            self.trade_count += 1;
        }

        self.points.push_back(point);
        self.trim_window();
    }

    /// Trim window based on window type
    fn trim_window(&mut self) {
        match self.window_type {
            WindowType::EventCount(max_events) => {
                while self.points.len() > max_events {
                    if let Some(removed) = self.points.pop_front() {
                        if let Some(vol) = removed.volume {
                            self.total_volume -= vol.as_f64();
                            self.trade_count -= 1;
                        }
                    }
                }
            }
            WindowType::VolumeCount(max_volume) => {
                while self.total_volume > max_volume as f64 && !self.points.is_empty() {
                    if let Some(removed) = self.points.pop_front() {
                        if let Some(vol) = removed.volume {
                            self.total_volume -= vol.as_f64();
                            self.trade_count -= 1;
                        }
                    }
                }
            }
            WindowType::TradeCount(max_trades) => {
                while self.trade_count > max_trades && !self.points.is_empty() {
                    if let Some(removed) = self.points.pop_front() {
                        if removed.volume.is_some() {
                            self.total_volume -= removed.volume.unwrap().as_f64();
                            self.trade_count -= 1;
                        }
                    }
                }
            }
            WindowType::ExponentialDecay { .. } => {
                // Keep all points for exponential weighting
                // But limit to reasonable size (e.g., 10000 events)
                while self.points.len() > 10000 {
                    self.points.pop_front();
                }
            }
        }
    }

    /// Calculate VWAP (Volume Weighted Average Price)
    pub fn vwap(&mut self) -> Option<f64> {
        if let Some(cached) = self.cached_vwap {
            return Some(cached);
        }

        match &self.window_type {
            WindowType::ExponentialDecay { half_life_events } => {
                // Exponentially weighted VWAP
                self.calculate_exponential_vwap(*half_life_events)
            }
            _ => {
                // Standard VWAP for other window types
                if self.points.is_empty() {
                    return None;
                }

                let mut total_value = 0.0;
                let mut total_volume = 0.0;

                for point in &self.points {
                    if let Some(volume) = point.volume {
                        let vol = volume.as_f64();
                        total_value += point.price.as_f64() * vol;
                        total_volume += vol;
                    }
                }

                if total_volume > 0.0 {
                    let vwap = total_value / total_volume;
                    self.cached_vwap = Some(vwap);
                    Some(vwap)
                } else {
                    None
                }
            }
        }
    }

    /// Calculate exponentially weighted VWAP
    fn calculate_exponential_vwap(&mut self, half_life_events: usize) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }

        let decay_factor = 0.5_f64.powf(1.0 / half_life_events as f64);
        let mut weighted_value = 0.0;
        let mut total_weight = 0.0;
        
        let current_event = self.event_counter;
        
        for point in &self.points {
            if let Some(volume) = point.volume {
                let events_ago = (current_event - point.event_number) as f64;
                let weight = decay_factor.powf(events_ago) * volume.as_f64();
                weighted_value += point.price.as_f64() * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            let vwap = weighted_value / total_weight;
            self.cached_vwap = Some(vwap);
            Some(vwap)
        } else {
            None
        }
    }

    /// Calculate rolling volatility (standard deviation of event-to-event returns)
    pub fn volatility(&mut self) -> Option<f64> {
        if let Some(cached) = self.cached_volatility {
            return Some(cached);
        }

        if self.points.len() < 2 {
            return None;
        }

        match &self.window_type {
            WindowType::ExponentialDecay { half_life_events } => {
                self.calculate_exponential_volatility(*half_life_events)
            }
            _ => {
                // Calculate event-to-event returns
                let mut returns = Vec::with_capacity(self.points.len() - 1);
                for i in 1..self.points.len() {
                    let prev_price = self.points[i - 1].price.as_f64();
                    let curr_price = self.points[i].price.as_f64();
                    if prev_price > 0.0 {
                        let ret = curr_price.ln() - prev_price.ln(); // Log returns
                        returns.push(ret);
                    }
                }

                if returns.is_empty() {
                    return None;
                }

                // Calculate standard deviation
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns.iter()
                    .map(|&r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                
                let volatility = variance.sqrt();
                self.cached_volatility = Some(volatility);
                Some(volatility)
            }
        }
    }

    /// Calculate exponentially weighted volatility
    fn calculate_exponential_volatility(&mut self, half_life_events: usize) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }

        let decay_factor = 0.5_f64.powf(1.0 / half_life_events as f64);
        let current_event = self.event_counter;
        
        // First pass: calculate weighted mean return
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for i in 1..self.points.len() {
            let prev_price = self.points[i - 1].price.as_f64();
            let curr_price = self.points[i].price.as_f64();
            if prev_price > 0.0 {
                let ret = curr_price.ln() - prev_price.ln();
                let events_ago = (current_event - self.points[i].event_number) as f64;
                let weight = decay_factor.powf(events_ago);
                weighted_sum += ret * weight;
                total_weight += weight;
            }
        }
        
        if total_weight == 0.0 {
            return None;
        }
        
        let weighted_mean = weighted_sum / total_weight;
        
        // Second pass: calculate weighted variance
        let mut weighted_var = 0.0;
        total_weight = 0.0;
        
        for i in 1..self.points.len() {
            let prev_price = self.points[i - 1].price.as_f64();
            let curr_price = self.points[i].price.as_f64();
            if prev_price > 0.0 {
                let ret = curr_price.ln() - prev_price.ln();
                let events_ago = (current_event - self.points[i].event_number) as f64;
                let weight = decay_factor.powf(events_ago);
                weighted_var += (ret - weighted_mean).powi(2) * weight;
                total_weight += weight;
            }
        }
        
        let volatility = (weighted_var / total_weight).sqrt();
        self.cached_volatility = Some(volatility);
        Some(volatility)
    }

    /// Get event-weighted moving average (EMA by event count)
    pub fn ema(&self) -> Option<f64> {
        match &self.window_type {
            WindowType::ExponentialDecay { half_life_events } => {
                if self.points.is_empty() {
                    return None;
                }
                
                let decay_factor = 0.5_f64.powf(1.0 / *half_life_events as f64);
                let current_event = self.event_counter;
                
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;
                
                for point in &self.points {
                    let events_ago = (current_event - point.event_number) as f64;
                    let weight = decay_factor.powf(events_ago);
                    weighted_sum += point.price.as_f64() * weight;
                    total_weight += weight;
                }
                
                if total_weight > 0.0 {
                    Some(weighted_sum / total_weight)
                } else {
                    None
                }
            }
            _ => {
                // For non-exponential windows, return simple average
                self.sma()
            }
        }
    }

    /// Get simple moving average over events
    pub fn sma(&self) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }

        let sum: f64 = self.points.iter()
            .map(|p| p.price.as_f64())
            .sum();
        
        Some(sum / self.points.len() as f64)
    }

    /// Get minimum price in window
    pub fn min(&self) -> Option<f64> {
        self.points.iter()
            .map(|p| p.price.as_f64())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Get maximum price in window
    pub fn max(&self) -> Option<f64> {
        self.points.iter()
            .map(|p| p.price.as_f64())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Get price momentum (current - first in window)
    pub fn momentum(&self) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }

        let first = self.points.front()?.price.as_f64();
        let last = self.points.back()?.price.as_f64();
        
        Some(last - first)
    }

    /// Get price momentum percentage
    pub fn momentum_pct(&self) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }

        let first = self.points.front()?.price.as_f64();
        let last = self.points.back()?.price.as_f64();
        
        if first > 0.0 {
            Some((last - first) / first * 100.0)
        } else {
            None
        }
    }
}

/// Event-based rolling features calculator
pub struct RollingFeatures {
    /// Short-term window (e.g., last 100 events)
    short_window: RollingWindow,
    /// Medium-term window (e.g., last 500 events)
    medium_window: RollingWindow,
    /// Long-term window (e.g., last 2000 events)
    long_window: RollingWindow,
    /// Volume-based window (e.g., last 1000 contracts)
    volume_window: RollingWindow,
    /// Trade-based window (e.g., last 50 trades)
    trade_window: RollingWindow,
    /// Volume accumulator
    volume_accumulator: f64,
    /// Trade count
    trade_count: u32,
}

impl RollingFeatures {
    pub fn new(_base_window_us: u64) -> Self {
        // Ignore time-based parameter, use event-based windows
        Self {
            short_window: RollingWindow::new(WindowType::EventCount(100)),
            medium_window: RollingWindow::new(WindowType::EventCount(500)),
            long_window: RollingWindow::new(WindowType::EventCount(2000)),
            volume_window: RollingWindow::new(WindowType::VolumeCount(1000)),
            trade_window: RollingWindow::new(WindowType::TradeCount(50)),
            volume_accumulator: 0.0,
            trade_count: 0,
        }
    }
    
    /// Create with custom window configurations
    pub fn with_windows(
        short_events: usize,
        medium_events: usize,
        long_events: usize,
        volume_contracts: u64,
        trade_count: usize,
    ) -> Self {
        Self {
            short_window: RollingWindow::new(WindowType::EventCount(short_events)),
            medium_window: RollingWindow::new(WindowType::EventCount(medium_events)),
            long_window: RollingWindow::new(WindowType::EventCount(long_events)),
            volume_window: RollingWindow::new(WindowType::VolumeCount(volume_contracts)),
            trade_window: RollingWindow::new(WindowType::TradeCount(trade_count)),
            volume_accumulator: 0.0,
            trade_count: 0,
        }
    }

    /// Update with new price data (called on every MBO event)
    pub fn update_price(&mut self, price: Price, timestamp: u64) {
        self.short_window.add_point(timestamp, price, None);
        self.medium_window.add_point(timestamp, price, None);
        self.long_window.add_point(timestamp, price, None);
        self.volume_window.add_point(timestamp, price, None);
        self.trade_window.add_point(timestamp, price, None);
    }

    /// Update with trade data (price and volume)
    pub fn update_trade(&mut self, price: Price, quantity: Quantity, timestamp: u64) {
        self.short_window.add_point(timestamp, price, Some(quantity));
        self.medium_window.add_point(timestamp, price, Some(quantity));
        self.long_window.add_point(timestamp, price, Some(quantity));
        self.volume_window.add_point(timestamp, price, Some(quantity));
        self.trade_window.add_point(timestamp, price, Some(quantity));
        
        self.volume_accumulator += quantity.as_f64();
        self.trade_count += 1;
    }

    /// Add features to feature vector
    pub fn add_to_vector(&mut self, features: &mut FeatureVector) {
        // VWAP features across different window types
        if let Some(vwap_short) = self.short_window.vwap() {
            features.add("vwap_short_events", vwap_short);
        }
        if let Some(vwap_medium) = self.medium_window.vwap() {
            features.add("vwap_medium_events", vwap_medium);
        }
        if let Some(vwap_long) = self.long_window.vwap() {
            features.add("vwap_long_events", vwap_long);
        }
        if let Some(vwap_volume) = self.volume_window.vwap() {
            features.add("vwap_volume_based", vwap_volume);
        }
        if let Some(vwap_trade) = self.trade_window.vwap() {
            features.add("vwap_trade_based", vwap_trade);
        }

        // Event-based volatility features
        if let Some(vol_short) = self.short_window.volatility() {
            features.add("volatility_short_events", vol_short);
        }
        if let Some(vol_medium) = self.medium_window.volatility() {
            features.add("volatility_medium_events", vol_medium);
        }
        if let Some(vol_long) = self.long_window.volatility() {
            features.add("volatility_long_events", vol_long);
        }

        // Event-based moving averages
        if let Some(sma_short) = self.short_window.sma() {
            features.add("sma_short_events", sma_short);
        }
        if let Some(sma_medium) = self.medium_window.sma() {
            features.add("sma_medium_events", sma_medium);
        }
        if let Some(sma_long) = self.long_window.sma() {
            features.add("sma_long_events", sma_long);
        }

        // Event-weighted moving averages
        if let Some(ema_short) = self.short_window.ema() {
            features.add("ema_short_events", ema_short);
        }

        // Price ranges over events
        if let (Some(min), Some(max)) = (self.short_window.min(), self.short_window.max()) {
            features.add("price_range_short_events", max - min);
            features.add("price_min_short_events", min);
            features.add("price_max_short_events", max);
        }

        // Event-based momentum
        if let Some(momentum) = self.short_window.momentum() {
            features.add("momentum_short_events", momentum);
        }
        if let Some(momentum_pct) = self.short_window.momentum_pct() {
            features.add("momentum_pct_short_events", momentum_pct);
        }
        if let Some(momentum_pct) = self.medium_window.momentum_pct() {
            features.add("momentum_pct_medium_events", momentum_pct);
        }

        // Volume features
        features.add("volume_accumulated", self.volume_accumulator);
        features.add("trade_count_total", self.trade_count as f64);
        if self.trade_count > 0 {
            features.add("avg_trade_size", self.volume_accumulator / self.trade_count as f64);
        }

        // Cross-window features (comparing different event horizons)
        if let (Some(sma_short), Some(sma_long)) = (self.short_window.sma(), self.long_window.sma()) {
            features.add("sma_ratio_short_long_events", sma_short / sma_long);
            features.add("sma_diff_short_long_events", sma_short - sma_long);
        }

        if let (Some(vol_short), Some(vol_long)) = (self.short_window.volatility(), self.long_window.volatility()) {
            if vol_long > 0.0 {
                features.add("volatility_ratio_short_long_events", vol_short / vol_long);
            }
        }

        // Volume vs event-based window comparison
        if let (Some(vwap_events), Some(vwap_volume)) = (self.short_window.vwap(), self.volume_window.vwap()) {
            features.add("vwap_events_vs_volume_ratio", vwap_events / vwap_volume);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_based_vwap() {
        let mut window = RollingWindow::new(WindowType::EventCount(5));
        
        // Add trades with volume
        window.add_point(1000, Price::from(100i64), Some(Quantity::from(10u32)));
        window.add_point(2000, Price::from(102i64), Some(Quantity::from(20u32)));
        window.add_point(3000, Price::from(101i64), Some(Quantity::from(30u32)));
        
        // VWAP = (100*10 + 102*20 + 101*30) / (10+20+30)
        //      = (1000 + 2040 + 3030) / 60
        //      = 6070 / 60 = 101.167
        let vwap = window.vwap().unwrap();
        assert!((vwap - 101.167).abs() < 0.001);
    }

    #[test]
    fn test_volume_based_window() {
        let mut window = RollingWindow::new(WindowType::VolumeCount(50));
        
        // Add trades totaling exactly 50 contracts
        window.add_point(1000, Price::from(100i64), Some(Quantity::from(20u32)));
        window.add_point(2000, Price::from(102i64), Some(Quantity::from(20u32)));
        window.add_point(3000, Price::from(104i64), Some(Quantity::from(10u32)));
        
        // Add one more trade that should push out the first
        window.add_point(4000, Price::from(106i64), Some(Quantity::from(40u32)));
        
        // Window should only contain last 50 contracts
        assert_eq!(window.total_volume, 50.0); // 10 + 40 = 50 (first two trades removed)
        assert_eq!(window.trade_count, 2); // First two trades removed
    }

    #[test]
    fn test_event_volatility() {
        let mut window = RollingWindow::new(WindowType::EventCount(4));
        
        // Add prices with event-to-event changes
        window.add_point(0, Price::from(100i64), None);
        window.add_point(1000, Price::from(102i64), None);
        window.add_point(2000, Price::from(98i64), None);
        window.add_point(3000, Price::from(101i64), None);
        
        let vol = window.volatility().unwrap();
        assert!(vol > 0.0); // Should have positive volatility
    }

    #[test]
    fn test_exponential_decay_window() {
        let mut window = RollingWindow::new(WindowType::ExponentialDecay { half_life_events: 2 });
        
        // Add several events
        for i in 0..5 {
            window.add_point(i * 1000, Price::from((100 + i) as i64), None);
        }
        
        // Recent events should have more weight in EMA
        let ema = window.ema().unwrap();
        assert!(ema > 102.0); // Should be weighted toward recent higher prices
    }

    #[test]
    fn test_event_window_trimming() {
        let mut window = RollingWindow::new(WindowType::EventCount(3));
        
        // Add more events than window size
        window.add_point(0, Price::from(100i64), None);
        window.add_point(1000, Price::from(101i64), None);
        window.add_point(2000, Price::from(102i64), None);
        window.add_point(3000, Price::from(103i64), None);
        window.add_point(4000, Price::from(104i64), None);
        
        // Should only keep last 3 events
        assert_eq!(window.points.len(), 3);
        assert_eq!(window.sma().unwrap(), 103.0); // (102 + 103 + 104) / 3
    }

    #[test]
    fn test_trade_count_window() {
        let mut window = RollingWindow::new(WindowType::TradeCount(2));
        
        // Add price updates without trades
        window.add_point(0, Price::from(100i64), None);
        window.add_point(1000, Price::from(101i64), None);
        
        // Add trades
        window.add_point(2000, Price::from(102i64), Some(Quantity::from(10u32)));
        window.add_point(3000, Price::from(103i64), Some(Quantity::from(20u32)));
        window.add_point(4000, Price::from(104i64), Some(Quantity::from(30u32)));
        
        // Should only keep last 2 trades
        assert_eq!(window.trade_count, 2);
        assert_eq!(window.total_volume, 50.0); // 20 + 30
    }
}