//! Feature collection and buffering
//!
//! This module handles efficient collection, storage and retrieval of features
//! for machine learning pipelines.

use crate::core::types::InstrumentId;
use std::collections::{HashMap, VecDeque};

/// Maximum number of feature vectors to keep in history
const DEFAULT_BUFFER_SIZE: usize = 10000;

/// A single feature value
#[derive(Debug, Clone)]
pub struct Feature {
    pub name: String,
    pub value: f64,
}

/// Feature vector representing all features at a specific point in time
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Instrument this feature vector belongs to
    pub instrument_id: InstrumentId,
    /// Timestamp when features were calculated
    pub timestamp: u64,
    /// Feature name to value mapping
    features: HashMap<String, f64>,
    /// Ordered list of feature names for consistent output
    feature_names: Vec<String>,
}

impl FeatureVector {
    /// Create a new feature vector
    pub fn new(instrument_id: InstrumentId, timestamp: u64) -> Self {
        Self {
            instrument_id,
            timestamp,
            features: HashMap::new(),
            feature_names: Vec::new(),
        }
    }

    /// Add a feature to the vector
    pub fn add(&mut self, name: &str, value: f64) {
        if !self.features.contains_key(name) {
            self.feature_names.push(name.to_string());
        }
        self.features.insert(name.to_string(), value);
    }

    /// Get a feature value
    pub fn get(&self, name: &str) -> Option<f64> {
        self.features.get(name).copied()
    }

    /// Get all feature names in order
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Get feature values in the same order as feature_names
    pub fn values(&self) -> Vec<f64> {
        self.feature_names
            .iter()
            .map(|name| self.features.get(name).copied().unwrap_or(0.0))
            .collect()
    }

    /// Convert to a dense array for ML frameworks
    pub fn to_array(&self) -> Vec<f64> {
        self.values()
    }

    /// Number of features
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// Circular buffer for feature vectors
pub struct FeatureBuffer {
    /// Maximum size of the buffer
    capacity: usize,
    /// Feature vectors stored in FIFO order
    buffer: VecDeque<FeatureVector>,
}

impl FeatureBuffer {
    /// Create a new feature buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    /// Add a feature vector to the buffer
    pub fn push(&mut self, features: FeatureVector) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(features);
    }

    /// Get the most recent feature vector
    pub fn latest(&self) -> Option<&FeatureVector> {
        self.buffer.back()
    }

    /// Get the last N feature vectors
    pub fn last_n(&self, n: usize) -> Vec<&FeatureVector> {
        let start = self.buffer.len().saturating_sub(n);
        self.buffer.range(start..).collect()
    }

    /// Get feature vectors within a time range
    pub fn range(&self, start: u64, end: u64) -> Vec<&FeatureVector> {
        self.buffer
            .iter()
            .filter(|fv| fv.timestamp >= start && fv.timestamp <= end)
            .collect()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Number of stored feature vectors
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Feature collector that manages buffers for multiple instruments
pub struct FeatureCollector {
    /// Buffers per instrument
    buffers: HashMap<InstrumentId, FeatureBuffer>,
    /// Default buffer capacity
    default_capacity: usize,
    /// All unique feature names seen
    all_feature_names: Vec<String>,
    /// Feature name to index mapping for consistent ordering
    feature_indices: HashMap<String, usize>,
}

impl FeatureCollector {
    /// Create a new feature collector
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BUFFER_SIZE)
    }

    /// Create with custom buffer capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffers: HashMap::new(),
            default_capacity: capacity,
            all_feature_names: Vec::new(),
            feature_indices: HashMap::new(),
        }
    }

    /// Collect a feature vector
    pub fn collect(&mut self, features: FeatureVector) {
        // Update global feature name tracking
        for name in features.feature_names() {
            if !self.feature_indices.contains_key(name) {
                let idx = self.all_feature_names.len();
                self.all_feature_names.push(name.clone());
                self.feature_indices.insert(name.clone(), idx);
            }
        }

        // Store in appropriate buffer
        let buffer = self.buffers
            .entry(features.instrument_id)
            .or_insert_with(|| FeatureBuffer::new(self.default_capacity));
        
        buffer.push(features);
    }

    /// Get the buffer for an instrument
    pub fn get_buffer(&self, instrument_id: InstrumentId) -> Option<&FeatureBuffer> {
        self.buffers.get(&instrument_id)
    }

    /// Get mutable buffer for an instrument
    pub fn get_buffer_mut(&mut self, instrument_id: InstrumentId) -> Option<&mut FeatureBuffer> {
        self.buffers.get_mut(&instrument_id)
    }

    /// Get all feature names in consistent order
    pub fn feature_names(&self) -> &[String] {
        &self.all_feature_names
    }

    /// Convert feature vectors to a 2D array with consistent feature ordering
    pub fn to_matrix(&self, instrument_id: InstrumentId, n_samples: Option<usize>) -> Option<Vec<Vec<f64>>> {
        let buffer = self.buffers.get(&instrument_id)?;
        
        let vectors: Vec<&FeatureVector> = if let Some(n) = n_samples {
            buffer.last_n(n)
        } else {
            buffer.buffer.iter().collect()
        };

        if vectors.is_empty() {
            return None;
        }

        // Convert to matrix with consistent feature ordering
        let matrix: Vec<Vec<f64>> = vectors.iter()
            .map(|fv| {
                self.all_feature_names.iter()
                    .map(|name| fv.get(name).unwrap_or(0.0))
                    .collect()
            })
            .collect();

        Some(matrix)
    }

    /// Get latest features for all instruments
    pub fn latest_all(&self) -> HashMap<InstrumentId, &FeatureVector> {
        self.buffers.iter()
            .filter_map(|(id, buffer)| {
                buffer.latest().map(|fv| (*id, fv))
            })
            .collect()
    }

    /// Total number of collected feature vectors
    pub fn total_vectors(&self) -> usize {
        self.buffers.values().map(|b| b.len()).sum()
    }

    /// Number of tracked instruments
    pub fn num_instruments(&self) -> usize {
        self.buffers.len()
    }

    /// Total number of unique features
    pub fn num_features(&self) -> usize {
        self.all_feature_names.len()
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Clear buffer for specific instrument
    pub fn clear_instrument(&mut self, instrument_id: InstrumentId) {
        if let Some(buffer) = self.buffers.get_mut(&instrument_id) {
            buffer.clear();
        }
    }

    /// Get buffer length for display
    pub fn len(&self) -> usize {
        self.total_vectors()
    }
}

impl Default for FeatureCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vector() {
        let mut fv = FeatureVector::new(1, 1000);
        
        fv.add("spread", 0.5);
        fv.add("volume", 100.0);
        fv.add("volatility", 0.02);
        
        assert_eq!(fv.len(), 3);
        assert_eq!(fv.get("spread"), Some(0.5));
        assert_eq!(fv.get("volume"), Some(100.0));
        
        let values = fv.values();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0], 0.5); // spread was added first
    }

    #[test]
    fn test_feature_buffer() {
        let mut buffer = FeatureBuffer::new(3);
        
        for i in 0..5 {
            let mut fv = FeatureVector::new(1, i * 1000);
            fv.add("price", 100.0 + i as f64);
            buffer.push(fv);
        }
        
        // Should only keep last 3
        assert_eq!(buffer.len(), 3);
        
        let latest = buffer.latest().unwrap();
        assert_eq!(latest.get("price"), Some(104.0));
        
        let last_2 = buffer.last_n(2);
        assert_eq!(last_2.len(), 2);
        assert_eq!(last_2[0].get("price"), Some(103.0));
        assert_eq!(last_2[1].get("price"), Some(104.0));
    }

    #[test]
    fn test_feature_collector() {
        let mut collector = FeatureCollector::with_capacity(10);
        
        // Add features for multiple instruments
        for i in 0..3 {
            let mut fv = FeatureVector::new(1, i * 1000);
            fv.add("feature_a", i as f64);
            fv.add("feature_b", i as f64 * 2.0);
            collector.collect(fv);
        }
        
        for i in 0..2 {
            let mut fv = FeatureVector::new(2, i * 1000);
            fv.add("feature_a", i as f64 * 10.0);
            fv.add("feature_c", i as f64 * 3.0);
            collector.collect(fv);
        }
        
        assert_eq!(collector.num_instruments(), 2);
        assert_eq!(collector.num_features(), 3); // feature_a, feature_b, feature_c
        assert_eq!(collector.total_vectors(), 5);
        
        // Check global feature ordering
        let feature_names = collector.feature_names();
        assert_eq!(feature_names.len(), 3);
        
        // Convert to matrix
        let matrix = collector.to_matrix(1, None).unwrap();
        assert_eq!(matrix.len(), 3); // 3 samples
        assert_eq!(matrix[0].len(), 3); // 3 features (with 0 for missing feature_c)
    }

    #[test]
    fn test_consistent_feature_ordering() {
        let mut collector = FeatureCollector::new();
        
        // Add features in different orders
        let mut fv1 = FeatureVector::new(1, 1000);
        fv1.add("a", 1.0);
        fv1.add("b", 2.0);
        collector.collect(fv1);
        
        let mut fv2 = FeatureVector::new(1, 2000);
        fv2.add("b", 3.0);
        fv2.add("c", 4.0);
        fv2.add("a", 5.0);
        collector.collect(fv2);
        
        let matrix = collector.to_matrix(1, None).unwrap();
        assert_eq!(matrix.len(), 2);
        
        // First vector: [1.0, 2.0, 0.0] (a, b, c)
        assert_eq!(matrix[0], vec![1.0, 2.0, 0.0]);
        
        // Second vector: [5.0, 3.0, 4.0] (a, b, c)
        assert_eq!(matrix[1], vec![5.0, 3.0, 4.0]);
    }
}