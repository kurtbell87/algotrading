//! Python bindings for core Rust types

#![allow(unsafe_op_in_unsafe_fn)]

use crate::core::Side;
use crate::core::types::{InstrumentId, Price, Quantity};
use crate::strategy::OrderSide;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Python wrapper for Price
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PyPrice {
    #[pyo3(get, set)]
    pub value: i64,
}

#[pymethods]
impl PyPrice {
    #[new]
    pub fn new(value: i64) -> Self {
        Self { value }
    }

    #[staticmethod]
    pub fn from_f64(value: f64) -> Self {
        Self {
            value: Price::from_f64(value).0,
        }
    }

    pub fn as_f64(&self) -> f64 {
        Price::new(self.value).as_f64()
    }

    pub fn __repr__(&self) -> String {
        format!("PyPrice({})", self.as_f64())
    }

    pub fn __str__(&self) -> String {
        format!("{:.4}", self.as_f64())
    }

    pub fn __add__(&self, other: &PyPrice) -> PyPrice {
        PyPrice::new(self.value + other.value)
    }

    pub fn __sub__(&self, other: &PyPrice) -> PyPrice {
        PyPrice::new(self.value - other.value)
    }

    pub fn __mul__(&self, scalar: f64) -> PyPrice {
        PyPrice::new((self.value as f64 * scalar) as i64)
    }

    pub fn __truediv__(&self, scalar: f64) -> PyPrice {
        PyPrice::new((self.value as f64 / scalar) as i64)
    }
}

impl From<Price> for PyPrice {
    fn from(price: Price) -> Self {
        Self { value: price.0 }
    }
}

impl From<PyPrice> for Price {
    fn from(py_price: PyPrice) -> Self {
        Price::new(py_price.value)
    }
}

/// Python wrapper for Quantity
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PyQuantity {
    #[pyo3(get, set)]
    pub value: u32,
}

#[pymethods]
impl PyQuantity {
    #[new]
    pub fn new(value: u32) -> Self {
        Self { value }
    }

    pub fn as_i64(&self) -> i64 {
        self.value as i64
    }

    pub fn as_u64(&self) -> u64 {
        self.value as u64
    }

    pub fn __repr__(&self) -> String {
        format!("PyQuantity({})", self.value)
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.value)
    }

    pub fn __add__(&self, other: &PyQuantity) -> PyQuantity {
        PyQuantity::new(self.value + other.value)
    }

    pub fn __sub__(&self, other: &PyQuantity) -> PyQuantity {
        PyQuantity::new(self.value.saturating_sub(other.value))
    }

    pub fn __mul__(&self, scalar: u32) -> PyQuantity {
        PyQuantity::new(self.value * scalar)
    }
}

impl From<Quantity> for PyQuantity {
    fn from(quantity: Quantity) -> Self {
        Self { value: quantity.0 }
    }
}

impl From<PyQuantity> for Quantity {
    fn from(py_quantity: PyQuantity) -> Self {
        Quantity::from(py_quantity.value)
    }
}

/// Python wrapper for Side
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PySide {
    Bid,
    Ask,
}

#[pymethods]
impl PySide {
    pub fn __repr__(&self) -> String {
        format!("PySide::{:?}", self)
    }

    pub fn __str__(&self) -> String {
        match self {
            PySide::Bid => "BID".to_string(),
            PySide::Ask => "ASK".to_string(),
        }
    }
}

impl From<Side> for PySide {
    fn from(side: Side) -> Self {
        match side {
            Side::Bid => PySide::Bid,
            Side::Ask => PySide::Ask,
        }
    }
}

impl From<PySide> for Side {
    fn from(py_side: PySide) -> Self {
        match py_side {
            PySide::Bid => Side::Bid,
            PySide::Ask => Side::Ask,
        }
    }
}

/// Python wrapper for OrderSide
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PyOrderSide {
    Buy,
    Sell,
    SellShort,
    BuyCover,
}

#[pymethods]
impl PyOrderSide {
    pub fn __repr__(&self) -> String {
        format!("PyOrderSide::{:?}", self)
    }

    pub fn __str__(&self) -> String {
        match self {
            PyOrderSide::Buy => "BUY".to_string(),
            PyOrderSide::Sell => "SELL".to_string(),
            PyOrderSide::SellShort => "SELL_SHORT".to_string(),
            PyOrderSide::BuyCover => "BUY_COVER".to_string(),
        }
    }
}

impl From<OrderSide> for PyOrderSide {
    fn from(side: OrderSide) -> Self {
        match side {
            OrderSide::Buy => PyOrderSide::Buy,
            OrderSide::Sell => PyOrderSide::Sell,
            OrderSide::SellShort => PyOrderSide::SellShort,
            OrderSide::BuyCover => PyOrderSide::BuyCover,
        }
    }
}

impl From<PyOrderSide> for OrderSide {
    fn from(py_side: PyOrderSide) -> Self {
        match py_side {
            PyOrderSide::Buy => OrderSide::Buy,
            PyOrderSide::Sell => OrderSide::Sell,
            PyOrderSide::SellShort => OrderSide::SellShort,
            PyOrderSide::BuyCover => OrderSide::BuyCover,
        }
    }
}

/// Python wrapper for market events
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyMarketEvent {
    #[pyo3(get)]
    pub event_type: String,
    #[pyo3(get)]
    pub instrument_id: InstrumentId,
    #[pyo3(get)]
    pub timestamp: u64,
    #[pyo3(get)]
    pub price: Option<PyPrice>,
    #[pyo3(get)]
    pub quantity: Option<PyQuantity>,
    #[pyo3(get)]
    pub side: Option<PySide>,
    #[pyo3(get)]
    pub bid_price: Option<PyPrice>,
    #[pyo3(get)]
    pub ask_price: Option<PyPrice>,
    #[pyo3(get)]
    pub bid_quantity: Option<PyQuantity>,
    #[pyo3(get)]
    pub ask_quantity: Option<PyQuantity>,
}

#[pymethods]
impl PyMarketEvent {
    #[new]
    pub fn new(event_type: String, instrument_id: InstrumentId, timestamp: u64) -> Self {
        Self {
            event_type,
            instrument_id,
            timestamp,
            price: None,
            quantity: None,
            side: None,
            bid_price: None,
            ask_price: None,
            bid_quantity: None,
            ask_quantity: None,
        }
    }

    #[staticmethod]
    pub fn with_trade(
        instrument_id: InstrumentId,
        timestamp: u64,
        price: PyPrice,
        quantity: PyQuantity,
        side: PySide,
    ) -> Self {
        Self {
            event_type: "Trade".to_string(),
            instrument_id,
            timestamp,
            price: Some(price),
            quantity: Some(quantity),
            side: Some(side),
            bid_price: None,
            ask_price: None,
            bid_quantity: None,
            ask_quantity: None,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (instrument_id, timestamp, bid_price=None, ask_price=None, bid_quantity=None, ask_quantity=None))]
    pub fn with_bbo(
        instrument_id: InstrumentId,
        timestamp: u64,
        bid_price: Option<PyPrice>,
        ask_price: Option<PyPrice>,
        bid_quantity: Option<PyQuantity>,
        ask_quantity: Option<PyQuantity>,
    ) -> Self {
        Self {
            event_type: "BBO".to_string(),
            instrument_id,
            timestamp,
            price: None,
            quantity: None,
            side: None,
            bid_price,
            ask_price,
            bid_quantity,
            ask_quantity,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyMarketEvent(type={}, instrument={}, timestamp={})",
            self.event_type, self.instrument_id, self.timestamp
        )
    }
}

/// Python wrapper for feature vectors
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyFeatureVector {
    #[pyo3(get)]
    pub features: HashMap<String, f64>,
    #[pyo3(get)]
    pub timestamp: u64,
}

#[pymethods]
impl PyFeatureVector {
    #[new]
    pub fn new(timestamp: u64) -> Self {
        Self {
            features: HashMap::new(),
            timestamp,
        }
    }

    pub fn add_feature(&mut self, name: String, value: f64) {
        self.features.insert(name, value);
    }

    pub fn get_feature(&self, name: &str) -> Option<f64> {
        self.features.get(name).copied()
    }

    pub fn get_features_as_array(&self) -> Vec<f64> {
        let mut keys: Vec<_> = self.features.keys().collect();
        keys.sort();
        keys.iter().map(|k| self.features[*k]).collect()
    }

    pub fn get_feature_names(&self) -> Vec<String> {
        let mut keys: Vec<_> = self.features.keys().cloned().collect();
        keys.sort();
        keys
    }

    pub fn __len__(&self) -> usize {
        self.features.len()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyFeatureVector(features={}, timestamp={})",
            self.features.len(),
            self.timestamp
        )
    }
}

/// Python wrapper for order requests
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyOrderRequest {
    #[pyo3(get)]
    pub strategy_id: String,
    #[pyo3(get)]
    pub instrument_id: InstrumentId,
    #[pyo3(get)]
    pub side: PyOrderSide,
    #[pyo3(get)]
    pub quantity: PyQuantity,
    #[pyo3(get)]
    pub price: Option<PyPrice>,
    #[pyo3(get)]
    pub order_type: String,
}

#[pymethods]
impl PyOrderRequest {
    #[new]
    pub fn new(
        strategy_id: String,
        instrument_id: InstrumentId,
        side: PyOrderSide,
        quantity: PyQuantity,
    ) -> Self {
        Self {
            strategy_id,
            instrument_id,
            side,
            quantity,
            price: None,
            order_type: "Market".to_string(),
        }
    }

    pub fn as_market_order(&mut self) {
        self.order_type = "Market".to_string();
        self.price = None;
    }

    pub fn as_limit_order(&mut self, price: PyPrice) {
        self.order_type = "Limit".to_string();
        self.price = Some(price);
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyOrderRequest({} {} {} @ {:?})",
            self.side.__str__(),
            self.quantity.__str__(),
            self.instrument_id,
            self.price
        )
    }
}

/// Python wrapper for strategy context
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyStrategyContext {
    #[pyo3(get)]
    pub strategy_id: String,
    #[pyo3(get)]
    pub current_time: u64,
    #[pyo3(get)]
    pub position_quantity: i64,
    #[pyo3(get)]
    pub position_pnl: f64,
    #[pyo3(get)]
    pub is_backtesting: bool,
}

#[pymethods]
impl PyStrategyContext {
    #[new]
    pub fn new(
        strategy_id: String,
        current_time: u64,
        position_quantity: i64,
        position_pnl: f64,
        is_backtesting: bool,
    ) -> Self {
        Self {
            strategy_id,
            current_time,
            position_quantity,
            position_pnl,
            is_backtesting,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyStrategyContext(id={}, position={}, pnl={:.2})",
            self.strategy_id, self.position_quantity, self.position_pnl
        )
    }
}

/// Python module for ML predictions
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyPrediction {
    #[pyo3(get)]
    pub signal: f64,
    #[pyo3(get)]
    pub confidence: f64,
    #[pyo3(get)]
    pub metadata: HashMap<String, f64>,
}

#[pymethods]
impl PyPrediction {
    #[new]
    pub fn new(signal: f64, confidence: f64) -> Self {
        Self {
            signal,
            confidence,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(&mut self, key: String, value: f64) {
        self.metadata.insert(key, value);
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyPrediction(signal={:.4}, confidence={:.4})",
            self.signal, self.confidence
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_price_operations() {
        let price1 = PyPrice::from_f64(100.50);
        let price2 = PyPrice::from_f64(99.75);

        let sum = price1.__add__(&price2);
        assert!((sum.as_f64() - 200.25).abs() < 0.01);

        let diff = price1.__sub__(&price2);
        assert!((diff.as_f64() - 0.75).abs() < 0.01);

        let scaled = price1.__mul__(2.0);
        assert!((scaled.as_f64() - 201.0).abs() < 0.01);
    }

    #[test]
    fn test_py_quantity_operations() {
        let qty1 = PyQuantity::new(100);
        let qty2 = PyQuantity::new(50);

        let sum = qty1.__add__(&qty2);
        assert_eq!(sum.value, 150);

        let diff = qty1.__sub__(&qty2);
        assert_eq!(diff.value, 50);

        let scaled = qty1.__mul__(2);
        assert_eq!(scaled.value, 200);
    }

    #[test]
    fn test_py_feature_vector() {
        let mut features = PyFeatureVector::new(1000);
        features.add_feature("price".to_string(), 100.0);
        features.add_feature("volume".to_string(), 1000.0);

        assert_eq!(features.__len__(), 2);
        assert_eq!(features.get_feature("price"), Some(100.0));
        assert_eq!(features.get_feature("nonexistent"), None);

        let array = features.get_features_as_array();
        assert_eq!(array.len(), 2);
    }

    #[test]
    fn test_py_order_request() {
        let order = PyOrderRequest::new(
            "test_strategy".to_string(),
            1,
            PyOrderSide::Buy,
            PyQuantity::new(100),
        );

        assert_eq!(order.strategy_id, "test_strategy");
        assert_eq!(order.instrument_id, 1);
        assert_eq!(order.side, PyOrderSide::Buy);
        assert_eq!(order.quantity.value, 100);
        assert_eq!(order.order_type, "Market");

        let mut limit_order = order;
        limit_order.as_limit_order(PyPrice::from_f64(100.0));
        assert_eq!(limit_order.order_type, "Limit");
        assert!(limit_order.price.is_some());
    }

    #[test]
    fn test_py_market_event() {
        let trade_event = PyMarketEvent::with_trade(
            1,
            1000,
            PyPrice::from_f64(100.0),
            PyQuantity::new(50),
            PySide::Bid,
        );

        assert_eq!(trade_event.event_type, "Trade");
        assert_eq!(trade_event.instrument_id, 1);
        assert_eq!(trade_event.timestamp, 1000);
        assert!(trade_event.price.is_some());
        assert_eq!(trade_event.price.unwrap().as_f64(), 100.0);

        let bbo_event = PyMarketEvent::with_bbo(
            1,
            2000,
            Some(PyPrice::from_f64(99.5)),
            Some(PyPrice::from_f64(100.5)),
            Some(PyQuantity::new(100)),
            Some(PyQuantity::new(200)),
        );

        assert_eq!(bbo_event.event_type, "BBO");
        assert!(bbo_event.bid_price.is_some());
        assert!(bbo_event.ask_price.is_some());
    }
}
