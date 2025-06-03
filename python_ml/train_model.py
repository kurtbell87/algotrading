#!/usr/bin/env python3
"""
Train a simple scikit-learn model for trading signal prediction.

This script demonstrates how to create and train ML models that can be
integrated with the Rust backtesting framework.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

def generate_synthetic_market_data(n_samples=10000):
    """Generate synthetic market data with features and labels"""
    np.random.seed(42)
    
    # Generate base price series (random walk)
    price_changes = np.random.normal(0, 0.01, n_samples)
    prices = 100 + np.cumsum(price_changes)
    
    # Create features
    data = {
        'mid_price': prices,
        'spread': np.random.uniform(0.01, 0.10, n_samples),
        'bid_size': np.random.exponential(1000, n_samples),
        'ask_size': np.random.exponential(1000, n_samples),
        'volume': np.random.exponential(5000, n_samples),
    }
    
    # Calculate derived features
    data['price_ma_5'] = pd.Series(prices).rolling(5).mean().fillna(prices[0])
    data['price_ma_20'] = pd.Series(prices).rolling(20).mean().fillna(prices[0])
    data['price_momentum'] = pd.Series(prices).pct_change(5).fillna(0)
    data['volume_imbalance'] = (data['bid_size'] - data['ask_size']) / (data['bid_size'] + data['ask_size'])
    data['relative_spread'] = data['spread'] / data['mid_price']
    
    # Create target variable (future price direction)
    # 1 = price will go up in next 10 periods, 0 = price will go down
    future_returns = pd.Series(prices).pct_change(10).shift(-10).fillna(0)
    labels = (future_returns > 0).astype(int)
    
    return pd.DataFrame(data), labels

def train_model():
    """Train a Random Forest model for trading signals"""
    print("Generating synthetic market data...")
    features_df, labels = generate_synthetic_market_data(10000)
    
    # Select features for model
    feature_columns = [
        'mid_price', 'spread', 'bid_size', 'ask_size', 'volume_imbalance'
    ]
    
    X = features_df[feature_columns].values
    y = labels.values
    
    # Remove any rows with NaN values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    print(f"Training data shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    print(f"Model accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nFeature Importances:")
    for feature, importance in zip(feature_columns, model.feature_importances_):
        print(f"  {feature}: {importance:.3f}")
    
    # Save model
    model_path = "trading_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature names and metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "feature_names": feature_columns,
        "accuracy": float(accuracy),
        "n_features": len(feature_columns),
        "training_samples": len(X_train),
        "description": "Random Forest model for predicting future price direction"
    }
    
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Model metadata saved to: model_metadata.json")
    
    return model, feature_columns, metadata

def test_model_predictions():
    """Test the saved model with some sample data"""
    print("\nTesting saved model...")
    
    # Load model and metadata
    model = joblib.load("trading_model.joblib")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    feature_names = metadata["feature_names"]
    
    # Create some test data
    test_samples = np.array([
        [100.0, 0.05, 1000, 1200, 0.091],  # Strong buy signal
        [100.0, 0.05, 1200, 1000, -0.091], # Strong sell signal
        [100.0, 0.05, 1000, 1000, 0.0],    # Neutral signal
    ])
    
    predictions = model.predict(test_samples)
    probabilities = model.predict_proba(test_samples)
    
    print(f"Test predictions: {predictions}")
    print("Prediction probabilities:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"  Sample {i+1}: {pred} (confidence: {prob.max():.3f})")
    
    return model, feature_names

if __name__ == "__main__":
    print("=== Scikit-learn ML Model Training for Trading ===\n")
    
    # Train the model
    model, feature_names, metadata = train_model()
    
    # Test the model
    test_model_predictions()
    
    print("\n=== Training Complete ===")
    print("Files created:")
    print("  - trading_model.joblib (trained model)")
    print("  - model_metadata.json (model information)")
    print("\nThe model can now be loaded in Rust using the Python bridge!")