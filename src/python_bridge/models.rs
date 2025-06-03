//! Python ML model integration for embedding ML models in strategies

use crate::python_bridge::types::{PyFeatureVector, PyPrediction};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;
use std::path::Path;

/// Error types for Python model operations
#[derive(Debug)]
pub enum PythonModelError {
    InitializationError(String),
    PredictionError(String),
    FeatureError(String),
    InvalidModel(String),
}

impl std::fmt::Display for PythonModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PythonModelError::InitializationError(msg) => {
                write!(f, "Model initialization error: {}", msg)
            }
            PythonModelError::PredictionError(msg) => write!(f, "Prediction error: {}", msg),
            PythonModelError::FeatureError(msg) => write!(f, "Feature error: {}", msg),
            PythonModelError::InvalidModel(msg) => write!(f, "Invalid model: {}", msg),
        }
    }
}

impl std::error::Error for PythonModelError {}

/// Trait for Python ML models that can be embedded in Rust strategies
pub trait PythonModel: Send + Sync {
    /// Make a prediction given feature vector
    fn predict(&self, features: &PyFeatureVector) -> Result<PyPrediction, PythonModelError>;

    /// Batch predict for multiple feature vectors
    fn predict_batch(
        &self,
        features: &[PyFeatureVector],
    ) -> Result<Vec<PyPrediction>, PythonModelError>;

    /// Get model metadata (name, version, etc.)
    fn get_metadata(&self) -> HashMap<String, String>;

    /// Check if model is ready for predictions
    fn is_ready(&self) -> bool;
}

/// Scikit-learn model wrapper
pub struct SklearnModel {
    model: PyObject,
    feature_names: Vec<String>,
    _model_type: String,
    metadata: HashMap<String, String>,
}

impl SklearnModel {
    /// Load a scikit-learn model from a pickle file
    pub fn from_pickle<P: AsRef<Path>>(
        model_path: P,
        feature_names: Vec<String>,
        model_type: String,
    ) -> Result<Self, PythonModelError> {
        Python::with_gil(|py| {
            // Import required modules
            let pickle = py.import_bound("pickle").map_err(|e| {
                PythonModelError::InitializationError(format!("Failed to import pickle: {}", e))
            })?;

            // Load the model
            let file = std::fs::File::open(&model_path).map_err(|e| {
                PythonModelError::InitializationError(format!("Failed to open model file: {}", e))
            })?;

            let model_bytes = std::io::Read::bytes(file)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    PythonModelError::InitializationError(format!(
                        "Failed to read model file: {}",
                        e
                    ))
                })?;

            let model = pickle.call_method1("loads", (model_bytes,)).map_err(|e| {
                PythonModelError::InitializationError(format!("Failed to unpickle model: {}", e))
            })?;

            // Verify model has predict method
            if !model.hasattr("predict").map_err(|e| {
                PythonModelError::InitializationError(format!(
                    "Error checking predict method: {}",
                    e
                ))
            })? {
                return Err(PythonModelError::InvalidModel(
                    "Model does not have predict method".to_string(),
                ));
            }

            let mut metadata = HashMap::new();
            metadata.insert(
                "model_path".to_string(),
                model_path.as_ref().to_string_lossy().to_string(),
            );
            metadata.insert("model_type".to_string(), model_type.clone());
            metadata.insert("feature_count".to_string(), feature_names.len().to_string());

            Ok(Self {
                model: model.to_object(py),
                feature_names,
                _model_type: model_type,
                metadata,
            })
        })
    }

    /// Create a new sklearn model from Python object
    pub fn from_python_object(
        model: PyObject,
        feature_names: Vec<String>,
        model_type: String,
    ) -> Result<Self, PythonModelError> {
        Python::with_gil(|py| {
            let model_ref = model.bind(py);

            // Verify model has predict method
            if !model_ref.hasattr("predict").map_err(|e| {
                PythonModelError::InitializationError(format!(
                    "Error checking predict method: {}",
                    e
                ))
            })? {
                return Err(PythonModelError::InvalidModel(
                    "Model does not have predict method".to_string(),
                ));
            }

            let mut metadata = HashMap::new();
            metadata.insert("model_type".to_string(), model_type.clone());
            metadata.insert("feature_count".to_string(), feature_names.len().to_string());

            Ok(Self {
                model,
                feature_names,
                _model_type: model_type,
                metadata,
            })
        })
    }

    /// Prepare features for prediction (convert to numpy array format expected by sklearn)
    fn prepare_features(&self, features: &PyFeatureVector) -> Result<PyObject, PythonModelError> {
        Python::with_gil(|py| {
            let numpy = py.import_bound("numpy").map_err(|e| {
                PythonModelError::FeatureError(format!("Failed to import numpy: {}", e))
            })?;

            // Create feature array in the correct order
            let mut feature_array = Vec::new();
            for feature_name in &self.feature_names {
                match features.get_feature(feature_name) {
                    Some(value) => feature_array.push(value),
                    None => {
                        return Err(PythonModelError::FeatureError(format!(
                            "Missing feature: {}",
                            feature_name
                        )));
                    }
                }
            }

            // Convert to numpy array with shape (1, n_features) for single prediction
            let py_list = PyList::new_bound(py, &[PyList::new_bound(py, feature_array)]);
            let np_array = numpy.call_method1("array", (&py_list,)).map_err(|e| {
                PythonModelError::FeatureError(format!("Failed to create numpy array: {}", e))
            })?;

            Ok(np_array.to_object(py))
        })
    }

    /// Prepare multiple feature vectors for batch prediction
    fn prepare_features_batch(
        &self,
        features: &[PyFeatureVector],
    ) -> Result<PyObject, PythonModelError> {
        Python::with_gil(|py| {
            let numpy = py.import_bound("numpy").map_err(|e| {
                PythonModelError::FeatureError(format!("Failed to import numpy: {}", e))
            })?;

            let mut feature_matrix = Vec::new();

            for feature_vec in features {
                let mut feature_array = Vec::new();
                for feature_name in &self.feature_names {
                    match feature_vec.get_feature(feature_name) {
                        Some(value) => feature_array.push(value),
                        None => {
                            return Err(PythonModelError::FeatureError(format!(
                                "Missing feature: {}",
                                feature_name
                            )));
                        }
                    }
                }
                feature_matrix.push(PyList::new_bound(py, feature_array));
            }

            // Convert to numpy array with shape (n_samples, n_features)
            let py_list = PyList::new_bound(py, feature_matrix);
            let np_array = numpy.call_method1("array", (&py_list,)).map_err(|e| {
                PythonModelError::FeatureError(format!("Failed to create numpy array: {}", e))
            })?;

            Ok(np_array.to_object(py))
        })
    }
}

impl PythonModel for SklearnModel {
    fn predict(&self, features: &PyFeatureVector) -> Result<PyPrediction, PythonModelError> {
        Python::with_gil(|py| {
            let model_ref = self.model.bind(py);
            let feature_array = self.prepare_features(features)?;

            // Make prediction
            let prediction = model_ref
                .call_method1("predict", (&feature_array,))
                .map_err(|e| {
                    PythonModelError::PredictionError(format!("Prediction failed: {}", e))
                })?;

            // Extract prediction value (assume single output for now)
            let pred_array = prediction.downcast::<PyList>().map_err(|e| {
                PythonModelError::PredictionError(format!(
                    "Failed to extract prediction array: {}",
                    e
                ))
            })?;

            let signal: f64 = pred_array
                .get_item(0)
                .map_err(|e| {
                    PythonModelError::PredictionError(format!(
                        "Failed to get prediction value: {}",
                        e
                    ))
                })?
                .extract()
                .map_err(|e| {
                    PythonModelError::PredictionError(format!(
                        "Failed to convert prediction to f64: {}",
                        e
                    ))
                })?;

            // Try to get prediction probabilities if available (for classification models)
            let confidence = if model_ref.hasattr("predict_proba").unwrap_or(false) {
                match model_ref.call_method1("predict_proba", (&feature_array,)) {
                    Ok(proba) => {
                        // Get max probability as confidence
                        let default_proba = PyList::new_bound(py, vec![0.5]);
                        let proba_array = proba.downcast::<PyList>().unwrap_or_else(|_| {
                            &default_proba // Default confidence
                        });

                        let probs: Vec<f64> = (0..proba_array.len())
                            .map(|i| {
                                proba_array
                                    .get_item(i)
                                    .ok()
                                    .and_then(|item| item.extract().ok())
                                    .unwrap_or(0.5)
                            })
                            .collect();

                        probs.iter().fold(0.0f64, |a, &b| a.max(b))
                    }
                    Err(_) => 0.5, // Default confidence if predict_proba fails
                }
            } else {
                0.5 // Default confidence for regression models
            };

            Ok(PyPrediction::new(signal, confidence))
        })
    }

    fn predict_batch(
        &self,
        features: &[PyFeatureVector],
    ) -> Result<Vec<PyPrediction>, PythonModelError> {
        if features.is_empty() {
            return Ok(Vec::new());
        }

        Python::with_gil(|py| {
            let model_ref = self.model.bind(py);
            let feature_matrix = self.prepare_features_batch(features)?;

            // Make batch prediction
            let predictions = model_ref
                .call_method1("predict", (&feature_matrix,))
                .map_err(|e| {
                    PythonModelError::PredictionError(format!("Batch prediction failed: {}", e))
                })?;

            // Extract prediction values
            let pred_array = predictions.downcast::<PyList>().map_err(|e| {
                PythonModelError::PredictionError(format!(
                    "Failed to extract prediction array: {}",
                    e
                ))
            })?;

            let mut results = Vec::new();

            for i in 0..pred_array.len() {
                if let Ok(item) = pred_array.get_item(i) {
                    let signal: f64 = item.extract().map_err(|e| {
                        PythonModelError::PredictionError(format!(
                            "Failed to convert prediction {} to f64: {}",
                            i, e
                        ))
                    })?;

                    // For batch predictions, use a default confidence
                    // In a real implementation, you might want to compute this properly
                    let confidence = 0.5;

                    results.push(PyPrediction::new(signal, confidence));
                }
            }

            Ok(results)
        })
    }

    fn get_metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }

    fn is_ready(&self) -> bool {
        Python::with_gil(|py| self.model.bind(py).hasattr("predict").unwrap_or(false))
    }
}

/// Factory for creating different types of Python models
pub struct PythonModelFactory;

impl PythonModelFactory {
    /// Create a scikit-learn model from pickle file
    pub fn create_sklearn_model<P: AsRef<Path>>(
        model_path: P,
        feature_names: Vec<String>,
        model_type: String,
    ) -> Result<Box<dyn PythonModel>, PythonModelError> {
        let model = SklearnModel::from_pickle(model_path, feature_names, model_type)?;
        Ok(Box::new(model))
    }

    /// Create a custom Python model from a Python object
    pub fn create_custom_model(
        model: PyObject,
        feature_names: Vec<String>,
        model_type: String,
    ) -> Result<Box<dyn PythonModel>, PythonModelError> {
        let sklearn_model = SklearnModel::from_python_object(model, feature_names, model_type)?;
        Ok(Box::new(sklearn_model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::python_bridge::types::PyFeatureVector;
    use std::collections::HashMap;

    #[test]
    fn test_feature_preparation() {
        // This test would require Python environment setup
        // In a real implementation, you'd want integration tests
        let mut features = PyFeatureVector::new(1000);
        features.add_feature("price".to_string(), 100.0);
        features.add_feature("volume".to_string(), 1000.0);

        assert_eq!(features.__len__(), 2);
        assert_eq!(features.get_feature("price"), Some(100.0));
    }

    #[test]
    fn test_model_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("model_type".to_string(), "RandomForest".to_string());
        metadata.insert("feature_count".to_string(), "5".to_string());

        assert_eq!(
            metadata.get("model_type"),
            Some(&"RandomForest".to_string())
        );
        assert_eq!(metadata.get("feature_count"), Some(&"5".to_string()));
    }

    #[test]
    fn test_prediction_struct() {
        let prediction = PyPrediction::new(0.75, 0.85);
        assert_eq!(prediction.signal, 0.75);
        assert_eq!(prediction.confidence, 0.85);

        let mut prediction_with_meta = prediction;
        prediction_with_meta.with_metadata("model_version".to_string(), 1.0);
        assert_eq!(
            prediction_with_meta.metadata.get("model_version"),
            Some(&1.0)
        );
    }
}
