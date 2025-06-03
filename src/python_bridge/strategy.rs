//! Python strategy integration and ML-enhanced strategies

use crate::core::types::{InstrumentId, Price, Quantity};
use crate::market_data::events::MarketEvent;
use crate::strategy::{Strategy, StrategyConfig, StrategyContext, StrategyOutput, StrategyError};
use crate::strategy::output::{OrderRequest, StrategyMetrics};
use crate::features::{FeatureExtractor, FeatureConfig};
use crate::python_bridge::models::PythonModel;
use crate::python_bridge::types::{PyFeatureVector, PyMarketEvent, PyStrategyContext, PyOrderRequest, PyPrediction, PyPrice, PyQuantity, PyOrderSide};
use pyo3::prelude::*;

/// ML-enhanced strategy that uses Python models for predictions
pub struct MLEnhancedStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// Feature extractor for creating ML inputs
    feature_extractor: FeatureExtractor,
    /// Python ML model
    model: Box<dyn PythonModel>,
    /// Model configuration
    model_config: MLModelConfig,
    /// Recent predictions for analysis
    recent_predictions: Vec<(u64, PyPrediction)>,
    /// Position tracking
    current_position: i64,
    /// Last prediction timestamp
    last_prediction_time: u64,
}

/// Configuration for ML model integration
#[derive(Debug, Clone)]
pub struct MLModelConfig {
    /// Prediction threshold for entry signals
    pub entry_threshold: f64,
    /// Prediction threshold for exit signals
    pub exit_threshold: f64,
    /// Confidence threshold for acting on predictions
    pub confidence_threshold: f64,
    /// Maximum position size
    pub max_position_size: i64,
    /// Order size per trade
    pub order_size: u32,
    /// Minimum time between predictions (microseconds)
    pub prediction_interval_us: u64,
    /// Features to extract for model input
    pub feature_names: Vec<String>,
    /// Whether to use limit or market orders
    pub use_limit_orders: bool,
    /// Limit order offset in ticks
    pub limit_order_offset_ticks: i64,
}

impl Default for MLModelConfig {
    fn default() -> Self {
        Self {
            entry_threshold: 0.6,
            exit_threshold: 0.4,
            confidence_threshold: 0.7,
            max_position_size: 10,
            order_size: 1,
            prediction_interval_us: 1_000_000, // 1 second
            feature_names: vec![
                "mid_price".to_string(),
                "spread".to_string(),
                "bid_size".to_string(),
                "ask_size".to_string(),
                "volume_imbalance".to_string(),
            ],
            use_limit_orders: false,
            limit_order_offset_ticks: 1,
        }
    }
}

impl MLEnhancedStrategy {
    /// Create a new ML-enhanced strategy
    pub fn new(
        strategy_id: String,
        instrument_id: InstrumentId,
        model: Box<dyn PythonModel>,
        model_config: MLModelConfig,
        feature_config: FeatureConfig,
    ) -> Result<Self, StrategyError> {
        let config = StrategyConfig::new(strategy_id, "ML Enhanced Strategy")
            .with_instrument(instrument_id)
            .with_max_position(model_config.max_position_size);
        
        let feature_extractor = FeatureExtractor::new(feature_config);
        
        Ok(Self {
            config,
            feature_extractor,
            model,
            model_config,
            recent_predictions: Vec::new(),
            current_position: 0,
            last_prediction_time: 0,
        })
    }
    
    /// Extract features and convert to Python format
    fn extract_python_features(&mut self, event: &MarketEvent, timestamp: u64) -> Result<PyFeatureVector, StrategyError> {
        // Convert market event to order book event for feature extraction
        // This is a simplified conversion - in practice you'd want more sophisticated mapping
        let order_book_event = match event {
            MarketEvent::BBO(bbo_event) => {
                crate::order_book::events::OrderBookEvent::BBOChanged {
                    instrument_id: bbo_event.instrument_id,
                    publisher_id: 0, // Default publisher ID
                    bid_price: bbo_event.bid_price,
                    ask_price: bbo_event.ask_price,
                    bid_quantity: bbo_event.bid_quantity,
                    ask_quantity: bbo_event.ask_quantity,
                    timestamp: bbo_event.timestamp,
                }
            }
            _ => return Err(StrategyError::FeatureError("Unsupported event type for features".to_string())),
        };
        
        // Extract features using the feature extractor
        self.feature_extractor.handle_event(&order_book_event);
        let feature_vector = self.feature_extractor.extract_features(order_book_event.instrument_id(), timestamp);
        
        // Convert to Python feature vector
        let mut py_features = PyFeatureVector::new(timestamp);
        
        for feature_name in &self.model_config.feature_names {
            if let Some(value) = feature_vector.get(feature_name) {
                py_features.add_feature(feature_name.clone(), value);
            } else {
                // Use default value for missing features
                py_features.add_feature(feature_name.clone(), 0.0);
            }
        }
        
        Ok(py_features)
    }
    
    /// Generate trading signal from ML prediction
    fn generate_signal(&self, prediction: &PyPrediction, current_position: i64) -> Option<PyOrderSide> {
        // Only act if confidence is high enough
        if prediction.confidence < self.model_config.confidence_threshold {
            return None;
        }
        
        // Generate entry signals
        if current_position == 0 {
            if prediction.signal > self.model_config.entry_threshold {
                return Some(PyOrderSide::Buy);
            } else if prediction.signal < -self.model_config.entry_threshold {
                return Some(PyOrderSide::SellShort);
            }
        }
        // Generate exit signals
        else if current_position > 0 && prediction.signal < -self.model_config.exit_threshold {
            return Some(PyOrderSide::Sell);
        } else if current_position < 0 && prediction.signal > self.model_config.exit_threshold {
            return Some(PyOrderSide::BuyCover);
        }
        
        None
    }
    
    /// Create order from signal
    fn create_order(
        &self,
        side: PyOrderSide,
        current_price: Price,
        context: &StrategyContext,
    ) -> Result<OrderRequest, StrategyError> {
        let instrument_id = self.config.instruments[0];
        let quantity = Quantity::from(self.model_config.order_size);
        
        let order = if self.model_config.use_limit_orders {
            let tick_size = 25; // TODO: Get from instrument config
            let offset = self.model_config.limit_order_offset_ticks * tick_size;
            
            let limit_price = match side {
                PyOrderSide::Buy | PyOrderSide::BuyCover => {
                    Price::new(current_price.0 - offset)
                }
                PyOrderSide::Sell | PyOrderSide::SellShort => {
                    Price::new(current_price.0 + offset)
                }
            };
            
            OrderRequest::limit_order(
                context.strategy_id.clone(),
                instrument_id,
                side.into(),
                limit_price,
                quantity,
            )
        } else {
            OrderRequest::market_order(
                context.strategy_id.clone(),
                instrument_id,
                side.into(),
                quantity,
            )
        };
        
        Ok(order)
    }
}

impl Strategy for MLEnhancedStrategy {
    fn initialize(&mut self, _context: &StrategyContext) -> Result<(), StrategyError> {
        self.recent_predictions.clear();
        self.current_position = 0;
        self.last_prediction_time = 0;
        Ok(())
    }
    
    fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput {
        let mut output = StrategyOutput::default();
        
        // Check if enough time has passed since last prediction
        if event.timestamp() - self.last_prediction_time < self.model_config.prediction_interval_us {
            return output;
        }
        
        // Extract current price for order creation
        let current_price = match event {
            MarketEvent::Trade(trade) => {
                if trade.instrument_id == self.config.instruments[0] {
                    Some(trade.price)
                } else {
                    None
                }
            }
            MarketEvent::BBO(bbo) => {
                if bbo.instrument_id == self.config.instruments[0] {
                    match (bbo.bid_price, bbo.ask_price) {
                        (Some(bid), Some(ask)) => {
                            Some(Price::from_f64((bid.as_f64() + ask.as_f64()) / 2.0))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        
        if let Some(price) = current_price {
            // Extract features for ML model
            match self.extract_python_features(event, event.timestamp()) {
                Ok(features) => {
                    // Make prediction
                    match self.model.predict(&features) {
                        Ok(prediction) => {
                            self.last_prediction_time = event.timestamp();
                            
                            // Store prediction for analysis
                            self.recent_predictions.push((event.timestamp(), prediction.clone()));
                            if self.recent_predictions.len() > 100 {
                                self.recent_predictions.remove(0);
                            }
                            
                            // Update position from context
                            self.current_position = context.position.quantity;
                            
                            // Generate trading signal
                            if let Some(side) = self.generate_signal(&prediction, self.current_position) {
                                match self.create_order(side, price, context) {
                                    Ok(order) => {
                                        output.orders.push(order);
                                    }
                                    Err(e) => {
                                        // Log error but continue
                                        eprintln!("Failed to create order: {}", e);
                                    }
                                }
                            }
                            
                            // Add metrics
                            let mut metrics = StrategyMetrics::new(event.timestamp());
                            metrics.add("ml_signal", prediction.signal);
                            metrics.add("ml_confidence", prediction.confidence);
                            metrics.add("current_position", self.current_position as f64);
                            metrics.add("prediction_count", self.recent_predictions.len() as f64);
                            output.set_metrics(metrics);
                        }
                        Err(e) => {
                            eprintln!("ML prediction failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Feature extraction failed: {}", e);
                }
            }
        }
        
        output
    }
    
    fn config(&self) -> &StrategyConfig {
        &self.config
    }
}

/// Python strategy wrapper that allows strategies to be written in Python
pub struct PythonStrategyWrapper {
    /// Strategy configuration
    config: StrategyConfig,
    /// Python strategy object
    py_strategy: PyObject,
    /// Whether the strategy is initialized
    initialized: bool,
}

impl PythonStrategyWrapper {
    /// Create a new Python strategy wrapper
    pub fn new(
        strategy_id: String,
        _instrument_id: InstrumentId,
        py_strategy: PyObject,
    ) -> Result<Self, StrategyError> {
        let config = StrategyConfig::new(strategy_id, "Python Strategy");
        
        // Verify Python strategy has required methods
        Python::with_gil(|py| {
            let strategy_ref = py_strategy.bind(py);
            
            if !strategy_ref.hasattr("on_market_event")? {
                return Err(StrategyError::InitializationError(
                    "Python strategy must have on_market_event method".to_string()
                ));
            }
            
            Ok(())
        })?;
        
        Ok(Self {
            config,
            py_strategy,
            initialized: false,
        })
    }
    
    /// Convert Rust market event to Python format
    fn convert_market_event(&self, event: &MarketEvent) -> PyMarketEvent {
        match event {
            MarketEvent::Trade(trade) => {
                PyMarketEvent::with_trade(
                    trade.instrument_id,
                    trade.timestamp,
                    PyPrice::from(trade.price),
                    PyQuantity::from(trade.quantity),
                    trade.aggressor_side.into(),
                )
            }
            MarketEvent::BBO(bbo) => {
                PyMarketEvent::with_bbo(
                    bbo.instrument_id,
                    bbo.timestamp,
                    bbo.bid_price.map(PyPrice::from),
                    bbo.ask_price.map(PyPrice::from),
                    bbo.bid_quantity.map(PyQuantity::from),
                    bbo.ask_quantity.map(PyQuantity::from),
                )
            }
            _ => {
                // Generic event for other types
                PyMarketEvent::new("Other".to_string(), 0, event.timestamp())
            }
        }
    }
    
    /// Convert Rust strategy context to Python format
    fn convert_context(&self, context: &StrategyContext) -> PyStrategyContext {
        PyStrategyContext::new(
            context.strategy_id.clone(),
            context.current_time,
            context.position.quantity,
            0.0, // TODO: Get actual P&L from position
            context.is_backtesting,
        )
    }
    
    /// Convert Python order requests to Rust format
    fn convert_order_request(&self, py_order: &PyOrderRequest) -> Result<OrderRequest, StrategyError> {
        let order = if py_order.order_type == "Market" {
            OrderRequest::market_order(
                py_order.strategy_id.clone(),
                py_order.instrument_id,
                py_order.side.into(),
                py_order.quantity.into(),
            )
        } else if py_order.order_type == "Limit" {
            if let Some(price) = py_order.price {
                OrderRequest::limit_order(
                    py_order.strategy_id.clone(),
                    py_order.instrument_id,
                    py_order.side.into(),
                    price.into(),
                    py_order.quantity.into(),
                )
            } else {
                return Err(StrategyError::InvalidOrder("Limit order missing price".to_string()));
            }
        } else {
            return Err(StrategyError::InvalidOrder(format!("Unknown order type: {}", py_order.order_type)));
        };
        
        Ok(order)
    }
}

impl Strategy for PythonStrategyWrapper {
    fn initialize(&mut self, context: &StrategyContext) -> Result<(), StrategyError> {
        Python::with_gil(|py| {
            let strategy_ref = self.py_strategy.bind(py);
            let py_context = self.convert_context(context);
            
            if strategy_ref.hasattr("initialize")? {
                strategy_ref.call_method1("initialize", (py_context,))?;
            }
            
            self.initialized = true;
            Ok(())
        })
        .map_err(|e: PyErr| StrategyError::InitializationError(format!("Python strategy initialization failed: {}", e)))
    }
    
    fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput {
        if !self.initialized {
            return StrategyOutput::default();
        }
        
        Python::with_gil(|py| {
            let strategy_ref = self.py_strategy.bind(py);
            let py_event = self.convert_market_event(event);
            let py_context = self.convert_context(context);
            
            match strategy_ref.call_method1("on_market_event", (py_event, py_context)) {
                Ok(result) => {
                    // Parse the result from Python
                    // Expected format: {"orders": [...], "metrics": {...}}
                    let mut output = StrategyOutput::default();
                    
                    if let Ok(dict) = result.downcast::<pyo3::types::PyDict>() {
                        // Extract orders
                        if let Ok(Some(orders_list)) = dict.get_item("orders") {
                            if let Ok(orders) = orders_list.downcast::<pyo3::types::PyList>() {
                                for order_item in orders.iter() {
                                    if let Ok(py_order) = order_item.extract::<PyOrderRequest>() {
                                        match self.convert_order_request(&py_order) {
                                            Ok(order) => output.orders.push(order),
                                            Err(e) => eprintln!("Failed to convert order: {}", e),
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Extract metrics
                        if let Ok(Some(metrics_dict)) = dict.get_item("metrics") {
                            if let Ok(metrics) = metrics_dict.downcast::<pyo3::types::PyDict>() {
                                let mut strategy_metrics = StrategyMetrics::new(event.timestamp());
                                for (key, value) in metrics.iter() {
                                    if let (Ok(k), Ok(v)) = (key.extract::<String>(), value.extract::<f64>()) {
                                        strategy_metrics.add(&k, v);
                                    }
                                }
                                output.set_metrics(strategy_metrics);
                            }
                        }
                    }
                    
                    output
                }
                Err(e) => {
                    eprintln!("Python strategy on_market_event failed: {}", e);
                    StrategyOutput::default()
                }
            }
        })
    }
    
    fn config(&self) -> &StrategyConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_model_config() {
        let config = MLModelConfig::default();
        assert_eq!(config.entry_threshold, 0.6);
        assert_eq!(config.confidence_threshold, 0.7);
        assert_eq!(config.feature_names.len(), 5);
        assert!(config.feature_names.contains(&"mid_price".to_string()));
    }

    #[test]
    fn test_signal_generation() {
        let config = MLModelConfig::default();
        
        // Create a mock strategy (without actual ML model)
        let prediction_buy = PyPrediction::new(0.8, 0.9); // Strong buy signal
        let prediction_sell = PyPrediction::new(-0.8, 0.9); // Strong sell signal
        let prediction_weak = PyPrediction::new(0.3, 0.5); // Weak signal, low confidence
        
        // Mock strategy for testing signal generation logic
        struct MockStrategy {
            config: MLModelConfig,
        }
        
        impl MockStrategy {
            fn generate_signal(&self, prediction: &PyPrediction, current_position: i64) -> Option<PyOrderSide> {
                if prediction.confidence < self.config.confidence_threshold {
                    return None;
                }
                
                if current_position == 0 {
                    if prediction.signal > self.config.entry_threshold {
                        return Some(PyOrderSide::Buy);
                    } else if prediction.signal < -self.config.entry_threshold {
                        return Some(PyOrderSide::SellShort);
                    }
                } else if current_position > 0 && prediction.signal < -self.config.exit_threshold {
                    return Some(PyOrderSide::Sell);
                } else if current_position < 0 && prediction.signal > self.config.exit_threshold {
                    return Some(PyOrderSide::BuyCover);
                }
                
                None
            }
        }
        
        let mock_strategy = MockStrategy { config };
        
        // Test entry signals
        assert_eq!(mock_strategy.generate_signal(&prediction_buy, 0), Some(PyOrderSide::Buy));
        assert_eq!(mock_strategy.generate_signal(&prediction_sell, 0), Some(PyOrderSide::SellShort));
        assert_eq!(mock_strategy.generate_signal(&prediction_weak, 0), None);
        
        // Test exit signals
        assert_eq!(mock_strategy.generate_signal(&prediction_sell, 5), Some(PyOrderSide::Sell));
        assert_eq!(mock_strategy.generate_signal(&prediction_buy, -5), Some(PyOrderSide::BuyCover));
    }

    #[test]
    fn test_feature_vector_conversion() {
        let mut py_features = PyFeatureVector::new(1000);
        py_features.add_feature("price".to_string(), 100.0);
        py_features.add_feature("volume".to_string(), 1000.0);
        py_features.add_feature("spread".to_string(), 0.05);
        
        assert_eq!(py_features.__len__(), 3);
        assert_eq!(py_features.get_feature("price"), Some(100.0));
        assert_eq!(py_features.get_feature("volume"), Some(1000.0));
        assert_eq!(py_features.get_feature("spread"), Some(0.05));
        
        let feature_array = py_features.get_features_as_array();
        assert_eq!(feature_array.len(), 3);
        
        let feature_names = py_features.get_feature_names();
        assert_eq!(feature_names.len(), 3);
        assert!(feature_names.contains(&"price".to_string()));
    }
}