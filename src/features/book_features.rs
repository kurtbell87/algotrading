//! Order book derived features for event-based trading
//!
//! This module extracts features from MBO events and order book state:
//! - Queue-reactive imbalance metrics
//! - Event-driven spread dynamics
//! - Order lifetime and modification patterns
//! - Book pressure from individual order placement/cancellation
//! - Queue position estimates

use crate::core::types::{Price, Quantity, OrderId};
use crate::order_book::events::OrderBookEvent;
use crate::features::collector::FeatureVector;
use std::collections::HashMap;

/// Book imbalance calculation modes
#[derive(Debug, Clone)]
pub enum ImbalanceMode {
    /// Simple volume imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    Simple,
    /// Weighted by distance from mid-price
    WeightedByDistance,
    /// Multi-level aggregated imbalance
    MultiLevel { levels: usize },
}

/// Spread metrics container
#[derive(Debug, Clone, Default)]
pub struct SpreadMetrics {
    /// Absolute spread: ask - bid
    pub absolute_spread: f64,
    /// Relative spread: (ask - bid) / mid_price
    pub relative_spread: f64,
    /// Effective spread considering volume
    pub effective_spread: Option<f64>,
    /// Quoted spread in basis points
    pub spread_bps: f64,
}

/// Book imbalance metrics
#[derive(Debug, Clone, Default)]
pub struct BookImbalance {
    /// Level 1 imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    pub level1_imbalance: f64,
    /// Multi-level imbalance (weighted average)
    pub multi_level_imbalance: Vec<f64>,
    /// Volume-weighted imbalance
    pub volume_weighted_imbalance: f64,
    /// Micro-price: mid + imbalance * spread/2
    pub micro_price: f64,
}

/// Book pressure metrics based on order events
#[derive(Debug, Clone, Default)]
pub struct BookPressure {
    /// Buy pressure: sum of bid volumes weighted by price proximity
    pub buy_pressure: f64,
    /// Sell pressure: sum of ask volumes weighted by price proximity
    pub sell_pressure: f64,
    /// Net pressure: buy_pressure - sell_pressure
    pub net_pressure: f64,
    /// Pressure ratio: buy_pressure / (buy_pressure + sell_pressure)
    pub pressure_ratio: f64,
    /// Order arrival pressure (new orders per event)
    pub arrival_pressure: f64,
    /// Cancellation pressure (cancelled volume / total volume)
    pub cancellation_pressure: f64,
}

/// Queue dynamics metrics
#[derive(Debug, Clone, Default)]
pub struct QueueMetrics {
    /// Estimated queue position for our hypothetical order at best bid
    pub bid_queue_position: f64,
    /// Estimated queue position for our hypothetical order at best ask
    pub ask_queue_position: f64,
    /// Average order lifetime in events
    pub avg_order_lifetime: f64,
    /// Order modification rate
    pub modification_rate: f64,
}

/// Order book features extractor
pub struct BookFeatures {
    /// Number of price levels to analyze
    #[allow(dead_code)]
    levels: usize,
    /// Latest spread metrics
    spread_metrics: SpreadMetrics,
    /// Latest imbalance metrics
    imbalance: BookImbalance,
    /// Latest pressure metrics
    pressure: BookPressure,
    /// Queue dynamics metrics
    queue_metrics: QueueMetrics,
    /// Track order lifetimes (order_id -> event_number when placed)
    order_birth_events: HashMap<OrderId, u64>,
    /// Current event counter
    event_counter: u64,
    /// Recent event counts for rate calculations
    recent_adds: u32,
    recent_cancels: u32,
    recent_modifies: u32,
    /// Total volume at best bid/ask for queue estimation
    best_bid_volume: f64,
    best_ask_volume: f64,
}

impl BookFeatures {
    pub fn new(levels: usize) -> Self {
        Self {
            levels,
            spread_metrics: SpreadMetrics::default(),
            imbalance: BookImbalance::default(),
            pressure: BookPressure::default(),
            queue_metrics: QueueMetrics::default(),
            order_birth_events: HashMap::new(),
            event_counter: 0,
            recent_adds: 0,
            recent_cancels: 0,
            recent_modifies: 0,
            best_bid_volume: 0.0,
            best_ask_volume: 0.0,
        }
    }

    /// Update features based on order book event
    pub fn update(&mut self, event: &OrderBookEvent) {
        self.event_counter += 1;
        
        match event {
            OrderBookEvent::OrderAdded { .. } => {
                self.recent_adds += 1;
                // Track order birth for lifetime calculation
                if let OrderBookEvent::OrderAdded { order_id, .. } = event {
                    self.order_birth_events.insert(*order_id, self.event_counter);
                }
            }
            OrderBookEvent::OrderCancelled { order_id, .. } => {
                self.recent_cancels += 1;
                // Calculate order lifetime
                if let Some(birth_event) = self.order_birth_events.remove(order_id) {
                    let lifetime = self.event_counter - birth_event;
                    self.update_order_lifetime(lifetime as f64);
                }
            }
            OrderBookEvent::OrderModified { .. } => {
                self.recent_modifies += 1;
            }
            _ => {}
        }
        
        // Update event-based pressure metrics
        self.update_event_pressure();
    }

    /// Update from BBO change event
    pub fn update_bbo(&mut self, bid_price: Option<Price>, bid_quantity: Option<Quantity>, 
                      ask_price: Option<Price>, ask_quantity: Option<Quantity>) {
        if let (Some(bp), Some(bq), Some(ap), Some(aq)) = (bid_price, bid_quantity, ask_price, ask_quantity) {
            self.update_spread_metrics(bp, bq, ap, aq);
            self.update_imbalance_metrics(bp, bq, ap, aq);
            self.update_queue_metrics(bq, aq);
            
            self.best_bid_volume = bq.0 as f64;
            self.best_ask_volume = aq.0 as f64;
        }
    }
    
    /// Update order lifetime moving average
    fn update_order_lifetime(&mut self, lifetime: f64) {
        // Exponential moving average
        let alpha = 0.1;
        self.queue_metrics.avg_order_lifetime = 
            alpha * lifetime + (1.0 - alpha) * self.queue_metrics.avg_order_lifetime;
    }
    
    /// Update event-based pressure metrics
    fn update_event_pressure(&mut self) {
        // Calculate arrival pressure (new orders per 100 events)
        if self.event_counter % 100 == 0 {
            self.pressure.arrival_pressure = self.recent_adds as f64;
            self.queue_metrics.modification_rate = self.recent_modifies as f64 / 100.0;
            
            // Reset counters
            self.recent_adds = 0;
            self.recent_cancels = 0;
            self.recent_modifies = 0;
        }
    }
    
    /// Calculate spread metrics
    fn update_spread_metrics(&mut self, bid_price: Price, bid_quantity: Quantity, 
                           ask_price: Price, ask_quantity: Quantity) {
        let bid_price = bid_price.as_f64();
        let ask_price = ask_price.as_f64();
        let mid_price = (bid_price + ask_price) / 2.0;
        
        self.spread_metrics.absolute_spread = ask_price - bid_price;
        self.spread_metrics.relative_spread = self.spread_metrics.absolute_spread / mid_price;
        self.spread_metrics.spread_bps = self.spread_metrics.relative_spread * 10000.0;
        
        // Effective spread considers execution at best prices
        let bid_vol = bid_quantity.0 as f64;
        let ask_vol = ask_quantity.0 as f64;
        let total_volume = bid_vol + ask_vol;
        if total_volume > 0.0 {
            let weighted_mid = (bid_price * ask_vol + ask_price * bid_vol) / total_volume;
            self.spread_metrics.effective_spread = Some(2.0 * (weighted_mid - mid_price).abs());
        }
    }

    /// Calculate imbalance metrics
    fn update_imbalance_metrics(&mut self, bid_price: Price, bid_quantity: Quantity,
                               ask_price: Price, ask_quantity: Quantity) {
        let bid_vol = bid_quantity.0 as f64;
        let ask_vol = ask_quantity.0 as f64;
        let total_vol = bid_vol + ask_vol;
        
        // Level 1 imbalance
        if total_vol > 0.0 {
            self.imbalance.level1_imbalance = (bid_vol - ask_vol) / total_vol;
        } else {
            self.imbalance.level1_imbalance = 0.0;
        }
        
        // Micro-price (volume-weighted mid price)
        let bid_price = bid_price.as_f64();
        let ask_price = ask_price.as_f64();
        if total_vol > 0.0 {
            self.imbalance.micro_price = (bid_price * ask_vol + ask_price * bid_vol) / total_vol;
        } else {
            self.imbalance.micro_price = (bid_price + ask_price) / 2.0;
        }
        
        // Multi-level imbalance would require full book data
        // For now, we'll use level 1 as a placeholder
        self.imbalance.multi_level_imbalance = vec![self.imbalance.level1_imbalance];
        self.imbalance.volume_weighted_imbalance = self.imbalance.level1_imbalance;
    }

    /// Update queue position estimates
    fn update_queue_metrics(&mut self, bid_quantity: Quantity, ask_quantity: Quantity) {
        // Simple queue position estimate: assume uniform distribution
        // In practice, this would use order placement patterns
        self.queue_metrics.bid_queue_position = bid_quantity.0 as f64 / 2.0;
        self.queue_metrics.ask_queue_position = ask_quantity.0 as f64 / 2.0;
    }

    /// Add features to feature vector
    pub fn add_to_vector(&self, features: &mut FeatureVector) {
        // Spread features
        features.add("spread_absolute", self.spread_metrics.absolute_spread);
        features.add("spread_relative", self.spread_metrics.relative_spread);
        features.add("spread_bps", self.spread_metrics.spread_bps);
        if let Some(eff_spread) = self.spread_metrics.effective_spread {
            features.add("spread_effective", eff_spread);
        }
        
        // Imbalance features
        features.add("imbalance_level1", self.imbalance.level1_imbalance);
        features.add("micro_price", self.imbalance.micro_price);
        features.add("imbalance_volume_weighted", self.imbalance.volume_weighted_imbalance);
        
        // Add multi-level imbalances
        for (i, &imb) in self.imbalance.multi_level_imbalance.iter().enumerate() {
            features.add(&format!("imbalance_level{}", i + 1), imb);
        }
        
        // Event-driven pressure features
        features.add("pressure_arrival", self.pressure.arrival_pressure);
        features.add("pressure_cancellation", self.pressure.cancellation_pressure);
        
        // Queue dynamics features
        features.add("queue_position_bid", self.queue_metrics.bid_queue_position);
        features.add("queue_position_ask", self.queue_metrics.ask_queue_position);
        features.add("order_lifetime_avg", self.queue_metrics.avg_order_lifetime);
        features.add("order_modification_rate", self.queue_metrics.modification_rate);
        
        // Event counts
        features.add("event_number", self.event_counter as f64);
    }

    /// Get spread metrics
    pub fn spread_metrics(&self) -> &SpreadMetrics {
        &self.spread_metrics
    }

    /// Get imbalance metrics
    pub fn imbalance(&self) -> &BookImbalance {
        &self.imbalance
    }

    /// Get pressure metrics
    pub fn pressure(&self) -> &BookPressure {
        &self.pressure
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Side;

    #[test]
    fn test_spread_calculation() {
        let mut features = BookFeatures::new(5);
        
        features.update_bbo(
            Some(Price::new(100_000_000_000)), // 100.0
            Some(Quantity::new(10)),
            Some(Price::new(101_000_000_000)), // 101.0
            Some(Quantity::new(15))
        );
        
        assert!((features.spread_metrics.absolute_spread - 1.0).abs() < 0.00001);
        assert!((features.spread_metrics.relative_spread - 0.00995).abs() < 0.00001);
        assert!((features.spread_metrics.spread_bps - 99.5).abs() < 0.01);
    }

    #[test]
    fn test_imbalance_calculation() {
        let mut features = BookFeatures::new(5);
        
        features.update_bbo(
            Some(Price::new(100_000_000_000)), // 100.0
            Some(Quantity::new(30)), // More buy volume
            Some(Price::new(101_000_000_000)), // 101.0
            Some(Quantity::new(10))  // Less sell volume
        );
        
        // Imbalance = (30 - 10) / (30 + 10) = 20/40 = 0.5
        assert_eq!(features.imbalance.level1_imbalance, 0.5);
        
        // Micro-price = (100 * 10 + 101 * 30) / 40 = 4030/40 = 100.75
        assert!((features.imbalance.micro_price - 100.75).abs() < 0.01);
    }

    #[test]
    fn test_order_lifetime_tracking() {
        let mut features = BookFeatures::new(5);
        
        // Add an order
        let order_id = 12345;
        let add_event = OrderBookEvent::OrderAdded {
            instrument_id: 1,
            publisher_id: 1,
            order_id,
            side: Side::Bid,
            price: Price::new(100_000_000_000),
            quantity: Quantity::new(10),
            timestamp: 1000,
        };
        features.update(&add_event);
        
        // Simulate 50 more events
        for _ in 0..50 {
            features.event_counter += 1;
        }
        
        // Cancel the order
        let cancel_event = OrderBookEvent::OrderCancelled {
            instrument_id: 1,
            publisher_id: 1,
            order_id,
            timestamp: 2000,
        };
        features.update(&cancel_event);
        
        // Should have recorded lifetime of ~50 events
        assert!(features.queue_metrics.avg_order_lifetime > 0.0);
    }

    #[test]
    fn test_event_pressure_metrics() {
        let mut features = BookFeatures::new(5);
        
        // Add 100 events to trigger pressure calculation
        for i in 0..100 {
            let event = OrderBookEvent::OrderAdded {
                instrument_id: 1,
                publisher_id: 1,
                order_id: i as u64,
                side: Side::Bid,
                price: Price::new(100_000_000_000),
                quantity: Quantity::new(10),
                timestamp: i as u64 * 1000,
            };
            features.update(&event);
        }
        
        // Should have calculated arrival pressure
        assert!(features.pressure.arrival_pressure > 0.0);
    }
}