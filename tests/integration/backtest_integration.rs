//! End-to-end integration tests for the backtesting system
//!
//! These tests verify the complete backtesting pipeline from data ingestion
//! through strategy execution to performance reporting.

use algotrading::core::types::{InstrumentId, Price, Quantity};
use algotrading::core::Side;
use algotrading::market_data::events::{MarketEvent, TradeEvent, BBOUpdate};
use algotrading::strategy::{Strategy, StrategyConfig, StrategyContext, StrategyOutput, StrategyError, OrderSide};
use algotrading::strategy::output::{OrderRequest, StrategyMetrics};
use algotrading::features::{FeatureExtractor, FeatureConfig};
use algotrading::backtest::engine::{BacktestEngine, BacktestConfig};
use algotrading::backtest::execution::{LatencyModel, FillModel};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use algotrading::strategies::{MarketMakerStrategy, market_maker::MarketMakerConfig};
use algotrading::strategies::{TrendFollowingStrategy, trend_following::TrendFollowingConfig};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Generate realistic synthetic market data for testing
fn generate_test_market_data(
    instrument_id: InstrumentId,
    num_events: usize,
    trend: MarketTrend,
) -> Vec<MarketEvent> {
    let mut events = Vec::with_capacity(num_events);
    let mut timestamp = 1_000_000; // Start at 1 second
    let mut price = 10000; // Starting price (100.00)
    let mut rng = SimpleRng::new(42);
    
    for i in 0..num_events {
        timestamp += rng.next_u64() % 1000 + 100; // 100-1100 microseconds between events
        
        // Generate price movement based on trend
        let change = match trend {
            MarketTrend::Bullish => ((rng.next_f64() - 0.3) * 10.0) as i64,
            MarketTrend::Bearish => ((rng.next_f64() - 0.7) * 10.0) as i64,
            MarketTrend::Sideways => ((rng.next_f64() - 0.5) * 5.0) as i64,
            MarketTrend::Volatile => ((rng.next_f64() - 0.5) * 20.0) as i64,
        };
        
        price = (price + change).max(5000).min(15000);
        
        // Mix of trade and BBO events
        if i % 3 == 0 {
            // BBO event
            let spread = 25 + (rng.next_u64() % 50) as i64; // 1-3 tick spread
            events.push(MarketEvent::BBO(BBOUpdate {
                instrument_id,
                bid_price: Some(Price::new(price - spread / 2)),
                ask_price: Some(Price::new(price + spread / 2)),
                bid_quantity: Some(Quantity::from((100 + rng.next_u32() % 400) as u32)),
                ask_quantity: Some(Quantity::from((100 + rng.next_u32() % 400) as u32)),
                bid_order_count: None,
                ask_order_count: None,
                timestamp,
            }));
        } else {
            // Trade event
            events.push(MarketEvent::Trade(TradeEvent {
                instrument_id,
                trade_id: i as u64,
                price: Price::new(price),
                quantity: Quantity::from((10 + rng.next_u32() % 90) as u32),
                aggressor_side: if rng.next_f64() > 0.5 { Side::Bid } else { Side::Ask },
                timestamp,
                buyer_order_id: None,
                seller_order_id: None,
            }));
        }
    }
    
    events
}

#[derive(Clone, Copy)]
enum MarketTrend {
    Bullish,
    Bearish,
    Sideways,
    Volatile,
}

/// Simple test strategy for integration testing
struct TestStrategy {
    config: StrategyConfig,
    trade_count: Arc<AtomicU64>,
    max_position: i64,
}

impl TestStrategy {
    fn new(strategy_id: String, instrument_id: InstrumentId, max_position: i64) -> Self {
        let config = StrategyConfig::new(strategy_id, "Test Strategy")
            .with_instrument(instrument_id)
            .with_max_position(max_position);
        
        Self {
            config,
            trade_count: Arc::new(AtomicU64::new(0)),
            max_position,
        }
    }
    
    fn get_trade_count(&self) -> u64 {
        self.trade_count.load(Ordering::Relaxed)
    }
}

impl Strategy for TestStrategy {
    fn initialize(&mut self, _context: &StrategyContext) -> Result<(), StrategyError> {
        Ok(())
    }
    
    fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput {
        let mut output = StrategyOutput::default();
        
        // Simple strategy: buy when position is 0, sell when max position reached
        if context.position.quantity == 0 && self.trade_count.load(Ordering::Relaxed) < 10 {
            if let MarketEvent::Trade(trade) = event {
                if trade.instrument_id == self.config.instruments[0] {
                    let order = OrderRequest::market_order(
                        context.strategy_id.clone(),
                        trade.instrument_id,
                        OrderSide::Buy,
                        Quantity::from(1u32),
                    );
                    output.orders.push(order);
                    self.trade_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else if context.position.quantity >= self.max_position {
            if let MarketEvent::Trade(trade) = event {
                if trade.instrument_id == self.config.instruments[0] {
                    let order = OrderRequest::market_order(
                        context.strategy_id.clone(),
                        trade.instrument_id,
                        OrderSide::Sell,
                        Quantity::from(context.position.quantity as u32),
                    );
                    output.orders.push(order);
                }
            }
        }
        
        // Add some metrics
        let mut metrics = StrategyMetrics::new(event.timestamp());
        metrics.add("position", context.position.quantity as f64);
        metrics.add("trades", self.trade_count.load(Ordering::Relaxed) as f64);
        output.set_metrics(metrics);
        
        output
    }
    
    fn config(&self) -> &StrategyConfig {
        &self.config
    }
}

#[test]
fn test_complete_backtest_pipeline() {
    println!("=== Complete Backtest Pipeline Test ===");
    
    // Create backtest configuration
    let config = BacktestConfig {
        start_time: Some(1_000_000),
        end_time: Some(10_000_000),
        initial_capital: 100_000.0,
        commission_per_contract: 0.5,
        latency_model: LatencyModel::Fixed(100), // 100 microseconds
        fill_model: FillModel::Optimistic,
        calculate_features: false,
        ..Default::default()
    };
    
    // Create backtest engine
    let mut engine = BacktestEngine::new(config.clone());
    
    // Add test strategy
    let test_strategy = TestStrategy::new("TestStrategy".to_string(), 1, 5);
    let trade_count_ref = test_strategy.trade_count.clone();
    engine.add_strategy(Box::new(test_strategy)).expect("Failed to add strategy");
    
    // Generate test data
    let events = generate_test_market_data(1, 1000, MarketTrend::Bullish);
    
    // Create mock data source
    let data_source = MockDataSource::new(events);
    
    // Run backtest
    let start = Instant::now();
    let report = engine.run_with_data_source(Box::new(data_source))
        .expect("Backtest failed");
    let elapsed = start.elapsed();
    
    println!("Backtest completed in {:?}", elapsed);
    println!("Events processed: {}", report.events_processed);
    println!("Trades executed: {}", trade_count_ref.load(Ordering::Relaxed));
    
    // Verify results
    assert!(report.events_processed > 0, "No events were processed");
    assert_eq!(report.strategy_results.len(), 1, "Strategy results missing");
    
    let strategy_result = &report.strategy_results[0];
    assert_eq!(strategy_result.strategy_id, "TestStrategy");
    assert_ne!(strategy_result.final_capital, config.initial_capital, "No trading occurred");
    
    // Check performance metrics
    assert!(report.performance_metrics.total_trades > 0, "No trades executed");
    
    println!("Test passed!");
}

#[test]
fn test_multiple_strategies_concurrent() {
    println!("=== Multiple Strategies Concurrent Test ===");
    
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission_per_contract: 0.5,
        ..Default::default()
    };
    
    let mut engine = BacktestEngine::new(config);
    
    // Add multiple strategies
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(MeanReversionStrategy::new(
            "MeanReversion1".to_string(),
            1,
            MeanReversionConfig::default(),
        )),
        Box::new(MarketMakerStrategy::new(
            "MarketMaker1".to_string(),
            1,
            MarketMakerConfig::default(),
        )),
        Box::new(TrendFollowingStrategy::new(
            "TrendFollower1".to_string(),
            1,
            TrendFollowingConfig::default(),
        )),
    ];
    
    for strategy in strategies {
        engine.add_strategy(strategy).expect("Failed to add strategy");
    }
    
    // Generate volatile market data
    let events = generate_test_market_data(1, 2000, MarketTrend::Volatile);
    let data_source = MockDataSource::new(events);
    
    // Run backtest
    let start = Instant::now();
    let report = engine.run_with_data_source(Box::new(data_source))
        .expect("Backtest failed");
    let elapsed = start.elapsed();
    
    println!("Multi-strategy backtest completed in {:?}", elapsed);
    println!("Events processed: {}", report.events_processed);
    
    // Verify all strategies produced results
    assert_eq!(report.strategy_results.len(), 3, "Not all strategies produced results");
    
    for result in &report.strategy_results {
        println!("Strategy: {}, Final P&L: {:.2}", 
                 result.strategy_id, 
                 result.total_pnl);
    }
    
    // Check portfolio-wide statistics
    assert!(report.portfolio_stats.total_positions > 0, "No positions taken");
    
    println!("Test passed!");
}

#[test]
fn test_risk_management_enforcement() {
    println!("=== Risk Management Enforcement Test ===");
    
    // Create a high-risk strategy that should trigger risk limits
    struct RiskyStrategy {
        config: StrategyConfig,
        order_count: u32,
    }
    
    impl RiskyStrategy {
        fn new() -> Self {
            let config = StrategyConfig::new("RiskyStrategy".to_string(), "Risky Strategy")
                .with_instrument(1)
                .with_max_position(10)
                .with_max_loss(Some(1000.0))
                .with_daily_max_loss(Some(500.0));
            
            Self { config, order_count: 0 }
        }
    }
    
    impl Strategy for RiskyStrategy {
        fn initialize(&mut self, _context: &StrategyContext) -> Result<(), StrategyError> {
            Ok(())
        }
        
        fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput {
            let mut output = StrategyOutput::default();
            
            // Try to place orders beyond risk limits
            if self.order_count < 20 {
                if let MarketEvent::Trade(trade) = event {
                    let order = OrderRequest::market_order(
                        context.strategy_id.clone(),
                        trade.instrument_id,
                        OrderSide::Buy,
                        Quantity::from(5u32), // Large position
                    );
                    output.orders.push(order);
                    self.order_count += 1;
                }
            }
            
            output
        }
        
        fn config(&self) -> &StrategyConfig {
            &self.config
        }
    }
    
    let config = BacktestConfig {
        initial_capital: 10_000.0, // Small capital for risk testing
        commission_per_contract: 1.0,
        ..Default::default()
    };
    
    let mut engine = BacktestEngine::new(config);
    engine.add_strategy(Box::new(RiskyStrategy::new())).expect("Failed to add strategy");
    
    // Generate bearish data to trigger losses
    let events = generate_test_market_data(1, 500, MarketTrend::Bearish);
    let data_source = MockDataSource::new(events);
    
    let report = engine.run_with_data_source(Box::new(data_source))
        .expect("Backtest failed");
    
    // Verify risk limits were enforced
    let result = &report.strategy_results[0];
    if let Some(stats) = &result.position_stats {
        assert!(stats.max_position <= 10, "Position limit exceeded");
        println!("Max position: {}", stats.max_position);
        println!("Total P&L: {:.2}", result.total_pnl);
    }
    
    println!("Risk management test passed!");
}

#[test]
fn test_order_lifecycle_and_fills() {
    println!("=== Order Lifecycle and Fills Test ===");
    
    // Strategy that tests different order types
    struct OrderTestStrategy {
        config: StrategyConfig,
        phase: u32,
    }
    
    impl OrderTestStrategy {
        fn new() -> Self {
            let config = StrategyConfig::new("OrderTest".to_string(), "Order Test Strategy")
                .with_instrument(1)
                .with_max_position(5);
            
            Self { config, phase: 0 }
        }
    }
    
    impl Strategy for OrderTestStrategy {
        fn initialize(&mut self, _context: &StrategyContext) -> Result<(), StrategyError> {
            self.phase = 0;
            Ok(())
        }
        
        fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput {
            let mut output = StrategyOutput::default();
            
            if let MarketEvent::Trade(trade) = event {
                match self.phase {
                    0 => {
                        // Test market order
                        let order = OrderRequest::market_order(
                            context.strategy_id.clone(),
                            trade.instrument_id,
                            OrderSide::Buy,
                            Quantity::from(2u32),
                        );
                        output.orders.push(order);
                        self.phase = 1;
                    }
                    1 if context.position.quantity > 0 => {
                        // Test limit order
                        let order = OrderRequest::limit_order(
                            context.strategy_id.clone(),
                            trade.instrument_id,
                            OrderSide::Sell,
                            Price::new(trade.price.0 + 50), // Above market
                            Quantity::from(1u32),
                        );
                        output.orders.push(order);
                        self.phase = 2;
                    }
                    _ => {}
                }
            }
            
            output
        }
        
        fn on_fill(&mut self, price: Price, quantity: i64, timestamp: u64, context: &StrategyContext) {
            println!("Fill received: {} @ {} at time {}", quantity, price.0, timestamp);
        }
        
        fn config(&self) -> &StrategyConfig {
            &self.config
        }
    }
    
    let config = BacktestConfig {
        initial_capital: 50_000.0,
        latency_model: LatencyModel::Variable { min: 50, max: 200 },
        fill_model: FillModel::Realistic {
            maker_fill_prob: 0.7,
            taker_slippage_ticks: 1,
        },
        ..Default::default()
    };
    
    let mut engine = BacktestEngine::new(config);
    engine.add_strategy(Box::new(OrderTestStrategy::new())).expect("Failed to add strategy");
    
    let events = generate_test_market_data(1, 200, MarketTrend::Sideways);
    let data_source = MockDataSource::new(events);
    
    let report = engine.run_with_data_source(Box::new(data_source))
        .expect("Backtest failed");
    
    // Verify orders were processed
    assert!(report.performance_metrics.total_trades > 0, "No trades executed");
    println!("Total trades: {}", report.performance_metrics.total_trades);
    
    // Check trade metrics
    if !report.trades.is_empty() {
        let first_trade = &report.trades[0];
        println!("First trade: {} contracts, P&L: {:.2}", 
                 first_trade.quantity, 
                 first_trade.pnl);
        assert!(first_trade.commission > 0.0, "Commission not calculated");
    }
    
    println!("Order lifecycle test passed!");
}

#[test]
fn test_performance_metrics_calculation() {
    println!("=== Performance Metrics Calculation Test ===");
    
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission_per_contract: 0.5,
        ..Default::default()
    };
    
    let mut engine = BacktestEngine::new(config);
    
    // Add a profitable strategy
    let strategy = MeanReversionStrategy::new(
        "ProfitableStrategy".to_string(),
        1,
        MeanReversionConfig {
            lookback_period: 10,
            entry_threshold: 1.5,
            exit_threshold: 0.5,
            order_size: 2,
            ..Default::default()
        },
    );
    
    engine.add_strategy(Box::new(strategy)).expect("Failed to add strategy");
    
    // Generate mean-reverting data
    let mut events = Vec::new();
    let mut timestamp = 1_000_000;
    let mut price = 10000;
    
    // Create mean-reverting pattern
    for i in 0..500 {
        timestamp += 1000;
        
        // Sine wave pattern for mean reversion
        let cycle = (i as f64 * 0.1).sin();
        price = 10000 + (cycle * 200.0) as i64;
        
        events.push(MarketEvent::Trade(TradeEvent {
            instrument_id: 1,
            trade_id: i,
            price: Price::new(price),
            quantity: Quantity::from(50u32),
            aggressor_side: if cycle > 0.0 { Side::Bid } else { Side::Ask },
            timestamp,
            buyer_order_id: None,
            seller_order_id: None,
        }));
    }
    
    let data_source = MockDataSource::new(events);
    let report = engine.run_with_data_source(Box::new(data_source))
        .expect("Backtest failed");
    
    // Verify metrics
    let metrics = &report.performance_metrics;
    println!("Performance Metrics:");
    println!("  Total trades: {}", metrics.total_trades);
    println!("  Win rate: {:.2}%", metrics.win_rate);
    println!("  Profit factor: {:.2}", metrics.profit_factor);
    println!("  Sharpe ratio: {:.3}", metrics.sharpe_ratio);
    println!("  Max drawdown: {:.2}%", metrics.max_drawdown);
    
    // Validate calculations
    if metrics.total_trades > 0 {
        assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 100.0, "Invalid win rate");
        assert!(metrics.winning_trades + metrics.losing_trades == metrics.total_trades, 
                "Trade count mismatch");
    }
    
    // Check equity curve
    assert!(!report.equity_curve.is_empty(), "No equity curve data");
    let final_equity = report.equity_curve.last().unwrap().equity;
    println!("Final equity: {:.2}", final_equity);
    
    println!("Performance metrics test passed!");
}

/// Mock data source for testing
struct MockDataSource {
    events: Vec<MarketEvent>,
    index: usize,
}

impl MockDataSource {
    fn new(events: Vec<MarketEvent>) -> Self {
        Self { events, index: 0 }
    }
}

impl algotrading::core::traits::MarketDataSource for MockDataSource {
    fn next_update(&mut self) -> Option<algotrading::core::MarketUpdate> {
        if self.index < self.events.len() {
            let event = self.events[self.index].clone();
            self.index += 1;
            
            // Convert MarketEvent to MarketUpdate
            match event {
                MarketEvent::Trade(trade) => {
                    Some(algotrading::core::MarketUpdate::Trade(algotrading::core::Trade {
                        instrument_id: trade.instrument_id,
                        price: trade.price,
                        quantity: trade.quantity,
                        side: trade.aggressor_side,
                        timestamp: trade.timestamp,
                    }))
                }
                _ => self.next_update(), // Skip non-trade events for simplicity
            }
        } else {
            None
        }
    }
}

/// Simple RNG for deterministic tests
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }
    
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}