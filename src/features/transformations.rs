//! Feature transformations for ML model preparation
//!
//! This module provides common feature transformations:
//! - Normalization (z-score, min-max)
//! - Feature engineering (polynomial, interactions)
//! - Dimensionality reduction preparation
//! - Missing value handling

use crate::features::FeatureVector;
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

    /// Transform a feature vector
    pub fn transform(&self, features: &mut FeatureVector) {
        match self.method {
            ScalingMethod::None => return,
            _ => {}
        }

        let feature_names = features.feature_names();
        let mut updates = Vec::new();
        
        for name in feature_names {
            if let Some(stats) = self.stats.get(name) {
                if let Some(value) = features.get(name) {
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
                    updates.push((name.to_string(), scaled));
                }
            }
        }
        
        // Apply updates
        for (name, value) in updates {
            features.add(&name, value);
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
                .filter_map(|name| features.get(name).map(|val| (name.to_string(), val)))
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
}