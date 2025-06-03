#!/usr/bin/env python3
"""
Example Python trading strategy that can be integrated with the Rust framework.

This demonstrates how to write trading strategies directly in Python
while leveraging the high-performance Rust backtesting engine.
"""

import joblib
import numpy as np
from typing import Dict, List, Optional

class PythonMLStrategy:
    """
    Python-based ML trading strategy that can be called from Rust.
    
    This strategy uses a trained scikit-learn model to generate trading signals
    based on market features.
    """
    
    def __init__(self, model_path: str = "trading_model.joblib", 
                 metadata_path: str = "model_metadata.json"):
        """Initialize the strategy with a trained model"""
        self.model = joblib.load(model_path)
        
        import json
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata["feature_names"]
        self.position = 0
        self.last_prediction_time = 0
        self.prediction_interval = 100_000  # 100ms in microseconds
        
        print(f"Python ML Strategy initialized with {len(self.feature_names)} features")
    
    def initialize(self, context):
        """Initialize strategy state"""
        self.position = 0
        self.last_prediction_time = 0
        print("Python ML Strategy initialized")
    
    def extract_features(self, event, context) -> Optional[Dict[str, float]]:
        """Extract features from market event"""
        if event.event_type == "Trade":
            return {
                "mid_price": float(event.price.as_f64() if event.price else 100.0),
                "spread": 0.05,  # Default spread
                "bid_size": 1000.0,  # Default size
                "ask_size": 1000.0,  # Default size
                "volume_imbalance": 0.0,  # Default imbalance
            }
        elif event.event_type == "BBO":
            if event.bid_price and event.ask_price:
                mid_price = (event.bid_price.as_f64() + event.ask_price.as_f64()) / 2.0
                spread = event.ask_price.as_f64() - event.bid_price.as_f64()
                
                bid_size = float(event.bid_quantity.value if event.bid_quantity else 1000)
                ask_size = float(event.ask_quantity.value if event.ask_quantity else 1000)
                volume_imbalance = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0.0
                
                return {
                    "mid_price": mid_price,
                    "spread": spread / mid_price,  # Relative spread
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "volume_imbalance": volume_imbalance,
                }
        
        return None
    
    def on_market_event(self, event, context):
        """Process market event and generate trading signals"""
        # Check if enough time has passed since last prediction
        if event.timestamp - self.last_prediction_time < self.prediction_interval:
            return {"orders": [], "metrics": {}}
        
        # Extract features
        features_dict = self.extract_features(event, context)
        if not features_dict:
            return {"orders": [], "metrics": {}}
        
        # Prepare feature array for model
        try:
            feature_array = np.array([[features_dict[name] for name in self.feature_names]])
        except KeyError as e:
            print(f"Missing feature: {e}")
            return {"orders": [], "metrics": {}}
        
        # Make prediction
        try:
            prediction = self.model.predict(feature_array)[0]
            probabilities = self.model.predict_proba(feature_array)[0]
            confidence = max(probabilities)
            
            self.last_prediction_time = event.timestamp
            
            # Generate trading signal
            orders = []
            current_position = context.position_quantity
            
            # Entry signals
            if current_position == 0 and confidence > 0.55:  # Confidence threshold
                if prediction == 1:  # Buy signal
                    orders.append({
                        "strategy_id": context.strategy_id,
                        "instrument_id": event.instrument_id,
                        "side": "Buy",
                        "quantity": {"value": 1},
                        "order_type": "Market",
                        "price": None
                    })
                elif prediction == 0 and confidence > 0.6:  # Stronger sell signal
                    orders.append({
                        "strategy_id": context.strategy_id,
                        "instrument_id": event.instrument_id,
                        "side": "SellShort",
                        "quantity": {"value": 1},
                        "order_type": "Market",
                        "price": None
                    })
            
            # Exit signals (when confidence is low)
            elif current_position != 0 and confidence < 0.52:
                exit_side = "Sell" if current_position > 0 else "BuyCover"
                orders.append({
                    "strategy_id": context.strategy_id,
                    "instrument_id": event.instrument_id,
                    "side": exit_side,
                    "quantity": {"value": abs(current_position)},
                    "order_type": "Market",
                    "price": None
                })
            
            # Return strategy output
            return {
                "orders": orders,
                "metrics": {
                    "ml_prediction": float(prediction),
                    "ml_confidence": float(confidence),
                    "current_position": float(current_position),
                    "mid_price": features_dict.get("mid_price", 0.0)
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"orders": [], "metrics": {}}

def test_python_strategy():
    """Test the Python strategy with sample data"""
    print("=== Testing Python ML Strategy ===")
    
    strategy = PythonMLStrategy()
    
    # Mock event and context objects
    class MockEvent:
        def __init__(self):
            self.event_type = "Trade"
            self.instrument_id = 1
            self.timestamp = 1_000_000
            self.price = MockPrice(100.0)
    
    class MockPrice:
        def __init__(self, value):
            self._value = value
        def as_f64(self):
            return self._value
    
    class MockContext:
        def __init__(self):
            self.strategy_id = "test_python_ml"
            self.position_quantity = 0
    
    # Test with sample event
    event = MockEvent()
    context = MockContext()
    
    # Initialize strategy
    strategy.initialize(context)
    
    # Process event
    result = strategy.on_market_event(event, context)
    
    print(f"Strategy output: {result}")
    print(f"Number of orders: {len(result['orders'])}")
    print(f"Metrics: {result['metrics']}")
    
    print("Python strategy test completed!")

if __name__ == "__main__":
    test_python_strategy()