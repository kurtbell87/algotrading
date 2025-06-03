//! Feature transformations for ML model preparation
//!
//! This module provides common feature transformations:
//! - Normalization (z-score, min-max)
//! - Feature engineering (polynomial, interactions)
//! - Dimensionality reduction preparation
//! - Missing value handling

use crate::features::collector::{FeatureCollector, FeatureVector};
use std::collections::HashMap;

/// Statistics for a single feature
#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

impl Default for FeatureStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
            min: f64::MAX,
            max: f64::MIN,
            count: 0,
        }
    }
}

/// Feature scaler for normalization
#[derive(Debug, Clone)]
pub struct FeatureScaler {
    /// Statistics per feature
    stats: HashMap<String, FeatureStats>,
    /// Scaling method
    method: ScalingMethod,
}

/// Scaling methods
#[derive(Debug, Clone, Copy)]
pub enum ScalingMethod {
    /// Z-score normalization: (x - mean) / std
    ZScore,
    /// Min-max scaling: (x - min) / (max - min)
    MinMax,
    /// Robust scaling: (x - median) / IQR
    Robust,
    /// No scaling
    None,
}

impl FeatureScaler {
    pub fn new(method: ScalingMethod) -> Self {
        Self {
            stats: HashMap::new(),
            method,
        }
    }

    /// Fit the scaler on a collection of feature vectors
    pub fn fit(&mut self, collector: &FeatureCollector, instrument_id: u32) {
        self.stats.clear();

        if let Some(buffer) = collector.get_buffer(instrument_id) {
            // First pass: collect values
            let mut feature_values: HashMap<String, Vec<f64>> = HashMap::new();

            for feature_vec in buffer.buffer.iter() {
                for name in feature_vec.feature_names() {
                    if let Some(value) = feature_vec.get(name) {
                        feature_values
                            .entry(name.clone())
                            .or_insert_with(Vec::new)
                            .push(value);
                    }
                }
            }

            // Calculate statistics
            for (name, values) in feature_values {
                if !values.is_empty() {
                    let stats = match self.method {
                        ScalingMethod::ZScore => self.calculate_zscore_stats(&values),
                        ScalingMethod::MinMax => self.calculate_minmax_stats(&values),
                        ScalingMethod::Robust => self.calculate_robust_stats(&values),
                        ScalingMethod::None => FeatureStats::default(),
                    };
                    self.stats.insert(name, stats);
                }
            }
        }
    }

    /// Calculate z-score statistics
    fn calculate_zscore_stats(&self, values: &[f64]) -> FeatureStats {
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;

        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count as f64;

        let std = variance.sqrt().max(1e-8); // Avoid division by zero

        FeatureStats {
            mean,
            std,
            min: values.iter().cloned().fold(f64::INFINITY, f64::min),
            max: values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            count,
        }
    }

    /// Calculate min-max statistics
    fn calculate_minmax_stats(&self, values: &[f64]) -> FeatureStats {
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        FeatureStats {
            mean: 0.0, // Not used for min-max
            std: 1.0,  // Not used for min-max
            min,
            max,
            count: values.len(),
        }
    }

    /// Calculate robust statistics (median and IQR)
    fn calculate_robust_stats(&self, values: &[f64]) -> FeatureStats {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let q1_idx = sorted.len() / 4;
        let q3_idx = 3 * sorted.len() / 4;
        let iqr = sorted[q3_idx] - sorted[q1_idx];

        FeatureStats {
            mean: median,       // Store median in mean field
            std: iqr.max(1e-8), // Store IQR in std field
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            count: values.len(),
        }
    }

    /// Transform a feature vector
    pub fn transform(&self, features: &mut FeatureVector) {
        match self.method {
            ScalingMethod::None => return,
            _ => {}
        }

        for name in features.feature_names().to_vec() {
            if let Some(stats) = self.stats.get(&name) {
                if let Some(value) = features.get(&name) {
                    let scaled = match self.method {
                        ScalingMethod::ZScore => (value - stats.mean) / stats.std,
                        ScalingMethod::MinMax => {
                            if stats.max > stats.min {
                                (value - stats.min) / (stats.max - stats.min)
                            } else {
                                0.5 // Default for constant features
                            }
                        }
                        ScalingMethod::Robust => (value - stats.mean) / stats.std,
                        ScalingMethod::None => value,
                    };
                    features.add(&name, scaled);
                }
            }
        }
    }

    /// Inverse transform a value
    pub fn inverse_transform(&self, feature_name: &str, scaled_value: f64) -> Option<f64> {
        let stats = self.stats.get(feature_name)?;

        let original = match self.method {
            ScalingMethod::ZScore => scaled_value * stats.std + stats.mean,
            ScalingMethod::MinMax => scaled_value * (stats.max - stats.min) + stats.min,
            ScalingMethod::Robust => scaled_value * stats.std + stats.mean,
            ScalingMethod::None => scaled_value,
        };

        Some(original)
    }
}

/// Feature engineering transformations
pub struct FeatureEngineer {
    /// Polynomial degree for polynomial features
    polynomial_degree: usize,
    /// Whether to include interaction features
    include_interactions: bool,
    /// Feature selection (names to include)
    selected_features: Option<Vec<String>>,
}

impl FeatureEngineer {
    pub fn new() -> Self {
        Self {
            polynomial_degree: 1,
            include_interactions: false,
            selected_features: None,
        }
    }

    /// Set polynomial degree
    pub fn with_polynomial(mut self, degree: usize) -> Self {
        self.polynomial_degree = degree;
        self
    }

    /// Enable interaction features
    pub fn with_interactions(mut self) -> Self {
        self.include_interactions = true;
        self
    }

    /// Select specific features
    pub fn with_selection(mut self, features: Vec<String>) -> Self {
        self.selected_features = Some(features);
        self
    }

    /// Apply feature engineering
    pub fn transform(&self, features: &mut FeatureVector) {
        // Get base features
        let base_features: Vec<(String, f64)> = if let Some(ref selected) = self.selected_features {
            selected
                .iter()
                .filter_map(|name| features.get(name).map(|val| (name.clone(), val)))
                .collect()
        } else {
            features
                .feature_names()
                .iter()
                .filter_map(|name| features.get(name).map(|val| (name.clone(), val)))
                .collect()
        };

        // Add polynomial features
        if self.polynomial_degree > 1 {
            for (name, value) in &base_features {
                for degree in 2..=self.polynomial_degree {
                    let poly_name = format!("{}_pow{}", name, degree);
                    features.add(&poly_name, value.powi(degree as i32));
                }
            }
        }

        // Add interaction features
        if self.include_interactions {
            for i in 0..base_features.len() {
                for j in i + 1..base_features.len() {
                    let (name1, val1) = &base_features[i];
                    let (name2, val2) = &base_features[j];
                    let interaction_name = format!("{}_{}_interact", name1, name2);
                    features.add(&interaction_name, val1 * val2);
                }
            }
        }

        // Add log transformations for positive features
        for (name, value) in &base_features {
            if *value > 0.0 {
                let log_name = format!("{}_log", name);
                features.add(&log_name, value.ln());
            }
        }
    }
}

/// Feature validator for data quality
pub struct FeatureValidator {
    /// Maximum allowed missing rate
    max_missing_rate: f64,
    /// Minimum variance required
    min_variance: f64,
    /// Features to always keep
    required_features: Vec<String>,
}

impl FeatureValidator {
    pub fn new() -> Self {
        Self {
            max_missing_rate: 0.1, // 10% missing allowed
            min_variance: 1e-8,    // Near-zero variance threshold
            required_features: vec![],
        }
    }

    /// Add required features
    pub fn with_required(mut self, features: Vec<String>) -> Self {
        self.required_features = features;
        self
    }

    /// Validate features in a collector
    pub fn validate(&self, collector: &FeatureCollector, instrument_id: u32) -> ValidationReport {
        let mut report = ValidationReport::default();

        // Check for required features first if no buffer exists
        if collector.get_buffer(instrument_id).is_none() {
            if !self.required_features.is_empty() {
                for required in &self.required_features {
                    report
                        .issues
                        .push(format!("Required feature '{}' is missing", required));
                }
                report.valid = false;
            } else {
                report
                    .issues
                    .push("No data available for instrument".to_string());
            }
            return report;
        }

        if let Some(buffer) = collector.get_buffer(instrument_id) {
            let total_samples = buffer.len();
            if total_samples == 0 {
                report.issues.push("No samples available".to_string());
                return report;
            }

            // Collect feature statistics
            let mut feature_counts: HashMap<String, usize> = HashMap::new();
            let mut feature_values: HashMap<String, Vec<f64>> = HashMap::new();

            for feature_vec in buffer.buffer.iter() {
                for name in feature_vec.feature_names() {
                    if let Some(value) = feature_vec.get(name) {
                        *feature_counts.entry(name.clone()).or_insert(0) += 1;
                        feature_values
                            .entry(name.clone())
                            .or_insert_with(Vec::new)
                            .push(value);
                    }
                }
            }

            // Check each feature
            for (name, count) in &feature_counts {
                let missing_rate = 1.0 - (*count as f64 / total_samples as f64);

                // Check missing rate
                if missing_rate > self.max_missing_rate && !self.required_features.contains(name) {
                    report.issues.push(format!(
                        "Feature '{}' has {:.1}% missing values",
                        name,
                        missing_rate * 100.0
                    ));
                    report.features_to_drop.push(name.clone());
                }

                // Check variance
                if let Some(values) = feature_values.get(name) {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;

                    if variance < self.min_variance && !self.required_features.contains(name) {
                        report
                            .issues
                            .push(format!("Feature '{}' has near-zero variance", name));
                        report.features_to_drop.push(name.clone());
                    }
                }
            }

            // Check for required features
            for required in &self.required_features {
                if !feature_counts.contains_key(required) {
                    report
                        .issues
                        .push(format!("Required feature '{}' is missing", required));
                }
            }

            report.valid = report.issues.is_empty();
        }

        report
    }
}

/// Validation report
#[derive(Debug, Default)]
pub struct ValidationReport {
    pub valid: bool,
    pub issues: Vec<String>,
    pub features_to_drop: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_features() -> FeatureVector {
        let mut features = FeatureVector::new(1, 1000);
        features.add("price", 100.0);
        features.add("volume", 50.0);
        features.add("spread", 0.5);
        features
    }

    #[test]
    fn test_zscore_scaling() {
        let mut collector = FeatureCollector::new();

        // Add some feature vectors
        for i in 0..10 {
            let mut fv = FeatureVector::new(1, i * 1000);
            fv.add("price", 100.0 + i as f64);
            fv.add("volume", 50.0 + i as f64 * 2.0);
            collector.collect(fv);
        }

        let mut scaler = FeatureScaler::new(ScalingMethod::ZScore);
        scaler.fit(&collector, 1);

        let mut test_features = create_test_features();
        scaler.transform(&mut test_features);

        // Price 100 should be below mean (104.5), so negative z-score
        assert!(test_features.get("price").unwrap() < 0.0);
    }

    #[test]
    fn test_minmax_scaling() {
        let mut collector = FeatureCollector::new();

        // Add features with known range
        for i in 0..=10 {
            let mut fv = FeatureVector::new(1, i * 1000);
            fv.add("price", 90.0 + i as f64 * 2.0); // Range: 90-110
            collector.collect(fv);
        }

        let mut scaler = FeatureScaler::new(ScalingMethod::MinMax);
        scaler.fit(&collector, 1);

        let mut test_features = FeatureVector::new(1, 1000);
        test_features.add("price", 100.0); // Middle of range
        scaler.transform(&mut test_features);

        // 100 is middle of 90-110, should scale to 0.5
        assert!((test_features.get("price").unwrap() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_polynomial_features() {
        let mut features = create_test_features();

        let engineer = FeatureEngineer::new()
            .with_polynomial(2)
            .with_selection(vec!["price".to_string(), "volume".to_string()]);

        engineer.transform(&mut features);

        // Check polynomial features
        assert_eq!(features.get("price_pow2"), Some(10000.0)); // 100^2
        assert_eq!(features.get("volume_pow2"), Some(2500.0)); // 50^2

        // Check log features
        assert!(features.get("price_log").is_some());
        assert!(features.get("volume_log").is_some());
    }

    #[test]
    fn test_interaction_features() {
        let mut features = create_test_features();

        let engineer = FeatureEngineer::new()
            .with_interactions()
            .with_selection(vec!["price".to_string(), "volume".to_string()]);

        engineer.transform(&mut features);

        // Check interaction
        assert_eq!(features.get("price_volume_interact"), Some(5000.0)); // 100 * 50
    }

    #[test]
    fn test_feature_validation() {
        let mut collector = FeatureCollector::new();

        // Add features with different characteristics
        for i in 0..10 {
            let mut fv = FeatureVector::new(1, i * 1000);
            fv.add("good_feature", 100.0 + i as f64);
            fv.add("constant_feature", 42.0); // No variance
            if i < 3 {
                fv.add("sparse_feature", i as f64); // Only 30% coverage
            }
            collector.collect(fv);
        }

        let validator = FeatureValidator::new();
        let report = validator.validate(&collector, 1);

        assert!(!report.valid);
        assert!(
            report
                .features_to_drop
                .contains(&"constant_feature".to_string())
        );
        assert!(
            report
                .features_to_drop
                .contains(&"sparse_feature".to_string())
        );
    }

    #[test]
    fn test_required_features() {
        let collector = FeatureCollector::new();

        let validator = FeatureValidator::new().with_required(vec!["critical_feature".to_string()]);

        let report = validator.validate(&collector, 1);

        assert!(!report.valid);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.contains("critical_feature"))
        );
    }
}
