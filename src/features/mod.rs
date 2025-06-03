//! Market microstructure feature engineering module
//!
//! This module provides feature extraction from order book and order flow data
//! for use in machine learning models and trading strategies.

pub mod book_features;
pub mod collector;
pub mod context_features;
pub mod flow_features;
pub mod rolling_features;
pub mod time_features;
pub mod transformations;

use crate::core::types::{InstrumentId, Price};
use crate::core::Side;
use crate::order_book::events::OrderBookEvent;
use std::collections::HashMap;

pub use book_features::{BookFeatures, BookImbalance, SpreadMetrics, BookPressure, QueueMetrics};
pub use collector::{FeatureCollector, FeatureVector, FeatureBuffer, Feature};
pub use context_features::{ContextFeatures, Position as FeaturePosition, RiskLimits};
pub use flow_features::{FlowFeatures, TradeFlowMetrics, AggressivePassiveMetrics, ArrivalRateMetrics, TradeSizeMetrics};
pub use rolling_features::{RollingFeatures, RollingWindow, WindowType};
pub use time_features::{TimeFeatures, TradingSession, SessionConfig};
pub use transformations::{FeatureScaler, ScalingMethod, FeatureEngineer, FeatureValidator, ValidationReport};

/// Configuration for feature extraction
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Number of price levels to analyze for book features
    pub book_levels: usize,
    /// Window size for rolling features (in microseconds)
    pub rolling_window_us: u64,
    /// Whether to calculate flow features
    pub enable_flow_features: bool,
    /// Whether to calculate rolling features
    pub enable_rolling_features: bool,
    /// Whether to calculate time-based features
    pub enable_time_features: bool,
    /// Whether to calculate context features
    pub enable_context_features: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            book_levels: 5,
            rolling_window_us: 60_000_000, // 1 minute
            enable_flow_features: true,
            enable_rolling_features: true,
            enable_time_features: true,
            enable_context_features: true,
        }
    }
}

/// Strategy context for feature extraction
pub struct StrategyContext {
    /// Current position
    pub position: FeaturePosition,
    /// Current market price
    pub current_price: Price,
    /// Risk limits
    pub risk_limits: RiskLimits,
}

/// Main feature extractor that coordinates all feature modules
pub struct FeatureExtractor {
    config: FeatureConfig,
    book_features: HashMap<InstrumentId, BookFeatures>,
    flow_features: HashMap<InstrumentId, FlowFeatures>,
    rolling_features: HashMap<InstrumentId, RollingFeatures>,
    time_features: Option<TimeFeatures>,
    context_features: HashMap<InstrumentId, ContextFeatures>,
    collector: FeatureCollector,
}

impl FeatureExtractor {
    pub fn new(config: FeatureConfig) -> Self {
        let time_features = if config.enable_time_features {
            Some(TimeFeatures::new(SessionConfig::default()))
        } else {
            None
        };
        
        Self {
            config: config.clone(),
            book_features: HashMap::new(),
            flow_features: HashMap::new(),
            rolling_features: HashMap::new(),
            time_features,
            context_features: HashMap::new(),
            collector: FeatureCollector::new(),
        }
    }

    /// Get or create book features for an instrument
    fn get_book_features(&mut self, instrument_id: InstrumentId) -> &mut BookFeatures {
        self.book_features
            .entry(instrument_id)
            .or_insert_with(|| BookFeatures::new(self.config.book_levels))
    }

    /// Get or create flow features for an instrument
    fn get_flow_features(&mut self, instrument_id: InstrumentId) -> &mut FlowFeatures {
        self.flow_features
            .entry(instrument_id)
            .or_insert_with(|| FlowFeatures::new())
    }

    /// Get or create rolling features for an instrument
    fn get_rolling_features(&mut self, instrument_id: InstrumentId) -> &mut RollingFeatures {
        self.rolling_features
            .entry(instrument_id)
            .or_insert_with(|| RollingFeatures::new(self.config.rolling_window_us))
    }
    
    /// Get or create context features for an instrument
    fn get_context_features(&mut self, instrument_id: InstrumentId, risk_limits: RiskLimits) -> &mut ContextFeatures {
        self.context_features
            .entry(instrument_id)
            .or_insert_with(|| ContextFeatures::new(risk_limits))
    }

    /// Extract all features for a given instrument at current state
    pub fn extract_features(&mut self, instrument_id: InstrumentId, timestamp: u64) -> FeatureVector {
        let mut features = FeatureVector::new(instrument_id, timestamp);

        // Extract book features
        if let Some(book_feat) = self.book_features.get(&instrument_id) {
            book_feat.add_to_vector(&mut features);
        }

        // Extract flow features
        if self.config.enable_flow_features {
            if let Some(flow_feat) = self.flow_features.get(&instrument_id) {
                flow_feat.add_to_vector(&mut features);
            }
        }

        // Extract rolling features
        if self.config.enable_rolling_features {
            if let Some(rolling_feat) = self.rolling_features.get_mut(&instrument_id) {
                rolling_feat.add_to_vector(&mut features);
            }
        }
        
        // Extract time features
        if self.config.enable_time_features {
            if let Some(ref time_feat) = self.time_features {
                time_feat.add_to_vector(&mut features, instrument_id, timestamp);
            }
        }

        features
    }
    
    /// Extract features with strategy context
    pub fn extract_with_context(
        &mut self,
        instrument_id: InstrumentId,
        timestamp: u64,
        context: &StrategyContext,
    ) -> FeatureVector {
        // Extract all standard features
        let mut features = self.extract_features(instrument_id, timestamp);
        
        // Add context features if enabled
        if self.config.enable_context_features {
            let context_feat = self.get_context_features(instrument_id, context.risk_limits.clone());
            
            // Update context features with current position
            context_feat.position = context.position.clone();
            
            // Add context features to vector
            context_feat.add_to_vector(&mut features, context.current_price, timestamp);
        }
        
        features
    }

    /// Get the feature collector for accessing historical features
    pub fn collector(&self) -> &FeatureCollector {
        &self.collector
    }

    /// Get mutable feature collector
    pub fn collector_mut(&mut self) -> &mut FeatureCollector {
        &mut self.collector
    }
    
    /// Get time features (if enabled)
    pub fn time_features(&self) -> Option<&TimeFeatures> {
        self.time_features.as_ref()
    }
    
    /// Get mutable time features
    pub fn time_features_mut(&mut self) -> Option<&mut TimeFeatures> {
        self.time_features.as_mut()
    }
    
    /// Update context features from a fill event
    pub fn update_context_from_fill(
        &mut self,
        instrument_id: InstrumentId,
        fill_price: Price,
        fill_quantity: i64,
        timestamp: u64,
        risk_limits: RiskLimits,
    ) {
        if self.config.enable_context_features {
            let context_feat = self.get_context_features(instrument_id, risk_limits);
            context_feat.update_position(fill_price, fill_quantity, timestamp);
        }
    }
}

impl FeatureExtractor {
    /// Process order book events and extract features
    pub fn handle_event(&mut self, event: &OrderBookEvent) {
        let (instrument_id, timestamp) = match event {
            OrderBookEvent::OrderAdded { instrument_id, timestamp, .. } |
            OrderBookEvent::OrderModified { instrument_id, timestamp, .. } |
            OrderBookEvent::OrderCancelled { instrument_id, timestamp, .. } |
            OrderBookEvent::BookCleared { instrument_id, timestamp, .. } |
            OrderBookEvent::BBOChanged { instrument_id, timestamp, .. } => (*instrument_id, *timestamp),
        };

        // Update book features with every MBO event
        let book_feat = self.get_book_features(instrument_id);
        book_feat.update(event);

        // Handle specific event types
        match event {
            OrderBookEvent::BBOChanged { 
                bid_price, 
                bid_quantity, 
                ask_price, 
                ask_quantity,
                .. 
            } => {
                // Update BBO-specific features
                book_feat.update_bbo(*bid_price, *bid_quantity, *ask_price, *ask_quantity);

                // Update rolling features with new price data
                if self.config.enable_rolling_features {
                    let rolling_feat = self.get_rolling_features(instrument_id);
                    if let (Some(bp), Some(ap)) = (bid_price, ask_price) {
                        let mid_price = Price::new((bp.0 + ap.0) / 2);
                        rolling_feat.update_price(mid_price, timestamp);
                    }
                }

                // Extract and collect features after BBO changes
                let feature_vector = self.extract_features(instrument_id, timestamp);
                self.collector.collect(feature_vector);
            }
            
            OrderBookEvent::OrderAdded { price, quantity, side, .. } => {
                // If this represents a trade (market order hitting resting order)
                // Update flow features
                if self.config.enable_flow_features {
                    let flow_feat = self.get_flow_features(instrument_id);
                    // Note: In real MBO, we'd need to distinguish market vs limit orders
                    // For now, treat aggressive orders as trades
                    let is_buy = matches!(side, Side::Bid);
                    flow_feat.update_trade(*price, *quantity, is_buy, timestamp);
                    
                    // Update rolling features with trade data
                    if self.config.enable_rolling_features {
                        let rolling_feat = self.get_rolling_features(instrument_id);
                        rolling_feat.update_trade(*price, *quantity, timestamp);
                    }
                }
            }
            
            _ => {} // Other events already handled in book features
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Quantity;

    #[test]
    fn test_feature_extractor_creation() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);
        assert!(extractor.book_features.is_empty());
        assert!(extractor.flow_features.is_empty());
        assert!(extractor.rolling_features.is_empty());
    }

    #[test]
    fn test_feature_extraction_on_bbo_change() {
        let mut extractor = FeatureExtractor::new(FeatureConfig::default());
        let instrument_id = 1;
        let timestamp = 1000;
        
        let event = OrderBookEvent::BBOChanged {
            instrument_id,
            publisher_id: 1,
            bid_price: Some(Price::new(100_000_000_000)),
            bid_quantity: Some(Quantity::new(10)),
            ask_price: Some(Price::new(101_000_000_000)),
            ask_quantity: Some(Quantity::new(15)),
            timestamp,
        };
        
        extractor.handle_event(&event);
        
        // Verify features were created
        assert!(extractor.book_features.contains_key(&instrument_id));
        assert!(extractor.rolling_features.contains_key(&instrument_id));
        
        // Verify features were collected
        assert_eq!(extractor.collector().len(), 1);
    }

    #[test] 
    fn test_order_event_processing() {
        let mut extractor = FeatureExtractor::new(FeatureConfig::default());
        let instrument_id = 1;
        
        // Add an order
        let add_event = OrderBookEvent::OrderAdded {
            instrument_id,
            publisher_id: 1,
            order_id: 12345,
            side: Side::Bid,
            price: Price::new(100_000_000_000),
            quantity: Quantity::new(10),
            timestamp: 1000,
        };
        
        extractor.handle_event(&add_event);
        
        // Verify book features were updated
        assert!(extractor.book_features.contains_key(&instrument_id));
        
        // If flow features are enabled, they should be updated too
        if extractor.config.enable_flow_features {
            assert!(extractor.flow_features.contains_key(&instrument_id));
        }
    }
}