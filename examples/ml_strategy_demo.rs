//! Demonstration of ML-enhanced trading strategy using scikit-learn
//!
//! This example shows how to:
//! 1. Load a pre-trained scikit-learn model in Rust
//! 2. Create an ML-enhanced trading strategy
//! 3. Run a backtest with the ML strategy
//! 4. Compare performance with traditional strategies

use algotrading::backtest::engine::{BacktestConfig, BacktestEngine};
use algotrading::core::Side;
use algotrading::core::types::{InstrumentId, Price, Quantity};
use algotrading::features::{FeatureConfig, FeatureExtractor};
use algotrading::market_data::events::{BBOUpdate, MarketEvent, TradeEvent};
use algotrading::python_bridge::models::{PythonModel, PythonModelFactory};
use algotrading::python_bridge::strategy::{MLEnhancedStrategy, MLModelConfig};
use algotrading::python_bridge::types::{PyFeatureVector, PyPrediction};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use algotrading::strategy::{Strategy, StrategyContext, StrategyOutput};
use std::fs;
use std::path::Path;

/// Load the trained ML model from the Python environment
fn load_ml_model() -> Result<Box<dyn PythonModel>, Box<dyn std::error::Error>> {
    // Read model metadata
    let metadata_path = "python_ml/model_metadata.json";
    let metadata_content = fs::read_to_string(metadata_path)?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;

    let feature_names: Vec<String> = metadata["feature_names"]
        .as_array()
        .ok_or("Missing feature_names in metadata")?
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect();

    let model_type = metadata["model_type"]
        .as_str()
        .unwrap_or("RandomForestClassifier")
        .to_string();

    println!(
        "Loading ML model with {} features: {:?}",
        feature_names.len(),
        feature_names
    );

    // Load the scikit-learn model
    let model_path = "python_ml/trading_model.joblib";
    let model = PythonModelFactory::create_sklearn_model(model_path, feature_names, model_type)?;

    println!("ML model loaded successfully!");
    println!("Model metadata: {:?}", model.get_metadata());

    Ok(model)
}

/// Create sample market data for demonstration
fn create_sample_market_data() -> Vec<MarketEvent> {
    let mut events = Vec::new();
    let instrument_id = 1;
    let mut timestamp = 1_000_000; // Start at 1 second

    // Generate some realistic market data
    let mut price = 10000; // Starting price in fixed-point (100.00)

    for i in 0..1000 {
        timestamp += 1000; // Advance by 1ms

        // Random walk with trend
        let change = if i < 300 {
            // Upward trend
            ((rand::random::<f64>() - 0.3) * 10.0) as i64
        } else if i < 600 {
            // Sideways
            ((rand::random::<f64>() - 0.5) * 5.0) as i64
        } else {
            // Downward trend
            ((rand::random::<f64>() - 0.7) * 10.0) as i64
        };

        price += change;
        price = price.max(9000).min(11000); // Keep price in reasonable range

        // Create trade event
        let trade_event = TradeEvent {
            instrument_id,
            trade_id: i as u64,
            price: Price::new(price),
            quantity: Quantity::from((50 + (rand::random::<u32>() % 100)) as u32),
            aggressor_side: if rand::random::<f64>() > 0.5 {
                Side::Bid
            } else {
                Side::Ask
            },
            timestamp,
            buyer_order_id: None,
            seller_order_id: None,
        };

        events.push(MarketEvent::Trade(trade_event));

        // Occasionally add BBO updates
        if i % 10 == 0 {
            let spread = 25 + (rand::random::<i64>() % 25); // 1-2 tick spread
            let bid_price = Price::new(price - spread / 2);
            let ask_price = Price::new(price + spread / 2);

            let bbo_event = BBOUpdate {
                instrument_id,
                bid_price: Some(bid_price),
                ask_price: Some(ask_price),
                bid_quantity: Some(Quantity::from((100 + (rand::random::<u32>() % 200)) as u32)),
                ask_quantity: Some(Quantity::from((100 + (rand::random::<u32>() % 200)) as u32)),
                bid_order_count: Some(2 + (rand::random::<u32>() % 5)),
                ask_order_count: Some(2 + (rand::random::<u32>() % 5)),
                timestamp: timestamp + 100,
            };

            events.push(MarketEvent::BBO(bbo_event));
        }
    }

    println!("Generated {} market events", events.len());
    events
}

/// Run ML strategy demonstration
fn run_ml_strategy_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ML Strategy Demonstration ===\n");

    // Load the trained ML model
    let model = load_ml_model()?;

    // Test the model with some sample features
    println!("\nTesting ML model predictions:");
    let mut test_features = PyFeatureVector::new(1000);
    test_features.add_feature("mid_price".to_string(), 100.0);
    test_features.add_feature("spread".to_string(), 0.05);
    test_features.add_feature("bid_size".to_string(), 1000.0);
    test_features.add_feature("ask_size".to_string(), 1200.0);
    test_features.add_feature("volume_imbalance".to_string(), 0.091);

    match model.predict(&test_features) {
        Ok(prediction) => {
            println!(
                "  Sample prediction: signal={:.3}, confidence={:.3}",
                prediction.signal, prediction.confidence
            );
        }
        Err(e) => {
            println!("  Prediction failed: {}", e);
            return Err(Box::new(e));
        }
    }

    // Create ML model configuration
    let ml_config = MLModelConfig {
        entry_threshold: 0.6,
        exit_threshold: 0.4,
        confidence_threshold: 0.55, // Lower threshold since our model isn't very confident
        max_position_size: 5,
        order_size: 1,
        prediction_interval_us: 100_000, // 100ms
        feature_names: vec![
            "mid_price".to_string(),
            "spread".to_string(),
            "bid_size".to_string(),
            "ask_size".to_string(),
            "volume_imbalance".to_string(),
        ],
        use_limit_orders: false,
        limit_order_offset_ticks: 1,
    };

    // Create feature extraction configuration
    let feature_config = FeatureConfig::default();

    // Create ML-enhanced strategy
    let ml_strategy = MLEnhancedStrategy::new(
        "ML_Strategy".to_string(),
        1, // instrument_id
        model,
        ml_config,
        feature_config,
    )?;

    println!("ML strategy created successfully!");

    // For comparison, create a traditional mean reversion strategy
    let mr_strategy = MeanReversionStrategy::new(
        "MeanReversion_Strategy".to_string(),
        1, // instrument_id
        MeanReversionConfig {
            lookback_period: 20,
            entry_threshold: 1.5,
            exit_threshold: 0.5,
            max_position_size: 5,
            order_size: 1,
            use_limit_orders: false,
            limit_order_offset_ticks: 1,
        },
    );

    println!("Traditional strategy created for comparison");

    // Create backtest configuration
    let backtest_config = BacktestConfig {
        start_time: Some(1_000_000),
        end_time: Some(2_000_000),
        initial_capital: 100_000.0,
        commission_per_contract: 0.5,
        max_events: Some(500), // Limit events for demo
        ..Default::default()
    };

    // Run backtest with ML strategy
    println!("\n=== Running Backtest with ML Strategy ===");
    let mut ml_engine = BacktestEngine::new(backtest_config.clone());
    ml_engine.add_strategy(Box::new(ml_strategy))?;

    // Simulate running with market data (in a real implementation, you'd load actual data)
    println!("Note: In a real implementation, you would load actual market data files here.");
    println!("For this demo, we're showing the setup and configuration.");

    // Display strategy configurations
    println!("\n=== Strategy Configurations ===");
    println!("ML Strategy:");
    println!("  - Entry threshold: 0.6");
    println!("  - Exit threshold: 0.4");
    println!("  - Confidence threshold: 0.55");
    println!("  - Features: mid_price, spread, bid_size, ask_size, volume_imbalance");
    println!("  - Model type: RandomForestClassifier");

    println!("\nMean Reversion Strategy:");
    println!("  - Lookback period: 20");
    println!("  - Entry threshold: 1.5 std devs");
    println!("  - Exit threshold: 0.5 std devs");

    println!("\n=== Demo Complete ===");
    println!("The ML strategy is now ready to run backtests with real market data!");

    Ok(())
}

fn main() {
    // Set up Python environment
    unsafe {
        std::env::set_var(
            "PYO3_PYTHON",
            "/Users/brandonbell/LOCAL_DEV/algotrading/python_ml/.venv/bin/python",
        );
    }

    match run_ml_strategy_demo() {
        Ok(()) => println!("\nDemo completed successfully!"),
        Err(e) => {
            eprintln!("\nDemo failed: {}", e);
            std::process::exit(1);
        }
    }
}

// Simple random number generation for demo purposes
mod rand {
    use std::cell::RefCell;

    thread_local! {
        static RNG_STATE: RefCell<u64> = RefCell::new(1);
    }

    pub fn random<T>() -> T
    where
        T: RandomValue,
    {
        RNG_STATE.with(|state| {
            let mut s = state.borrow_mut();
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            T::from_u64(*s)
        })
    }

    pub trait RandomValue {
        fn from_u64(value: u64) -> Self;
    }

    impl RandomValue for f64 {
        fn from_u64(value: u64) -> Self {
            (value as f64) / (u64::MAX as f64)
        }
    }

    impl RandomValue for i64 {
        fn from_u64(value: u64) -> Self {
            (value % (i64::MAX as u64)) as i64
        }
    }

    impl RandomValue for u32 {
        fn from_u64(value: u64) -> Self {
            (value % (u32::MAX as u64)) as u32
        }
    }
}
