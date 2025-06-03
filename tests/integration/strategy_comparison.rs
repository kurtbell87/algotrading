//! Strategy comparison framework for evaluating different trading strategies
//!
//! This module provides tools for comparing multiple strategies on the same
//! market data to evaluate relative performance.

use algotrading::core::types::{InstrumentId, Price, Quantity};
use algotrading::core::Side;
use algotrading::market_data::events::{MarketEvent, TradeEvent, BBOUpdate};
use algotrading::strategy::Strategy;
use algotrading::strategies::{
    MeanReversionStrategy, mean_reversion::MeanReversionConfig,
    MarketMakerStrategy, market_maker::MarketMakerConfig,
    TrendFollowingStrategy, trend_following::TrendFollowingConfig,
};
use algotrading::backtest::engine::{BacktestEngine, BacktestConfig};
use algotrading::backtest::execution::{LatencyModel, FillModel};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Results from a strategy comparison
#[derive(Debug)]
struct ComparisonResult {
    strategy_name: String,
    total_pnl: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
    total_trades: usize,
    profit_factor: f64,
    avg_trade_duration: f64,
}

/// Compare multiple strategies on the same market data
fn compare_strategies(
    strategies: Vec<(String, Box<dyn Strategy>)>,
    market_data: Vec<MarketEvent>,
    config: BacktestConfig,
) -> Vec<ComparisonResult> {
    let mut results = Vec::new();
    
    for (name, strategy) in strategies {
        println!("Running backtest for: {}", name);
        
        let mut engine = BacktestEngine::new(config.clone());
        engine.add_strategy(strategy).expect("Failed to add strategy");
        
        let data_source = MockDataSource::new(market_data.clone());
        let report = engine.run_with_data_source(Box::new(data_source))
            .expect("Backtest failed");
        
        // Extract results
        if let Some(strategy_result) = report.strategy_results.first() {
            let metrics = &report.performance_metrics;
            
            results.push(ComparisonResult {
                strategy_name: name,
                total_pnl: strategy_result.total_pnl,
                sharpe_ratio: metrics.sharpe_ratio,
                max_drawdown: metrics.max_drawdown,
                win_rate: metrics.win_rate,
                total_trades: metrics.total_trades,
                profit_factor: metrics.profit_factor,
                avg_trade_duration: metrics.avg_trade_duration,
            });
        }
    }
    
    results
}

#[test]
fn test_strategy_comparison_framework() {
    println!("=== Strategy Comparison Framework Test ===\n");
    
    // Create market data with different regimes
    let market_data = create_multi_regime_market_data();
    
    // Setup strategies with different configurations
    let strategies: Vec<(String, Box<dyn Strategy>)> = vec![
        // Conservative Mean Reversion
        ("MR_Conservative".to_string(), Box::new(MeanReversionStrategy::new(
            "MR_Conservative".to_string(),
            1,
            MeanReversionConfig {
                lookback_period: 50,
                entry_threshold: 2.0,
                exit_threshold: 0.5,
                max_position_size: 3,
                order_size: 1,
                ..Default::default()
            },
        ))),
        
        // Aggressive Mean Reversion
        ("MR_Aggressive".to_string(), Box::new(MeanReversionStrategy::new(
            "MR_Aggressive".to_string(),
            1,
            MeanReversionConfig {
                lookback_period: 20,
                entry_threshold: 1.0,
                exit_threshold: 0.3,
                max_position_size: 5,
                order_size: 2,
                ..Default::default()
            },
        ))),
        
        // Market Maker
        ("MarketMaker".to_string(), Box::new(MarketMakerStrategy::new(
            "MarketMaker".to_string(),
            1,
            MarketMakerConfig {
                spread_multiple: 1.5,
                inventory_target: 0,
                max_position: 10,
                order_size: 1,
                min_spread_bps: 5,
                max_spread_bps: 50,
                update_threshold_bps: 10,
                volatility_lookback: 100,
                ..Default::default()
            },
        ))),
        
        // Trend Following
        ("TrendFollower".to_string(), Box::new(TrendFollowingStrategy::new(
            "TrendFollower".to_string(),
            1,
            TrendFollowingConfig {
                fast_ma_period: 10,
                slow_ma_period: 30,
                atr_period: 14,
                atr_multiplier: 2.0,
                momentum_threshold: 0.001,
                max_position_size: 5,
                order_size: 1,
                use_volume_confirmation: false,
                volume_ma_period: 20,
                ..Default::default()
            },
        ))),
    ];
    
    // Run comparison with realistic settings
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission_per_contract: 0.5,
        latency_model: LatencyModel::Variable { min: 50, max: 150 },
        fill_model: FillModel::Realistic {
            maker_fill_prob: 0.7,
            taker_slippage_ticks: 1,
        },
        ..Default::default()
    };
    
    let results = compare_strategies(strategies, market_data, config);
    
    // Display comparison results
    println!("\n=== Strategy Comparison Results ===\n");
    println!("{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
             "Strategy", "Total P&L", "Sharpe", "Max DD%", "Win Rate%", "Trades", "PF");
    println!("{:-<104}", "");
    
    for result in &results {
        println!("{:<20} {:>12.2} {:>12.3} {:>12.1} {:>12.1} {:>12} {:>12.2}",
                 result.strategy_name,
                 result.total_pnl,
                 result.sharpe_ratio,
                 result.max_drawdown,
                 result.win_rate,
                 result.total_trades,
                 result.profit_factor);
    }
    
    // Rank strategies
    let mut ranked = results;
    ranked.sort_by(|a, b| b.sharpe_ratio.partial_cmp(&a.sharpe_ratio).unwrap());
    
    println!("\n=== Strategy Rankings (by Sharpe Ratio) ===");
    for (rank, result) in ranked.iter().enumerate() {
        println!("{}. {} (Sharpe: {:.3}, P&L: ${:.2})",
                 rank + 1,
                 result.strategy_name,
                 result.sharpe_ratio,
                 result.total_pnl);
    }
    
    println!("\nStrategy comparison test completed!");
}

#[test]
fn test_regime_specific_performance() {
    println!("=== Regime-Specific Performance Test ===\n");
    
    // Test strategies in different market regimes
    let regimes = vec![
        ("Trending_Up", create_trending_market(true, 0.7)),
        ("Trending_Down", create_trending_market(false, 0.7)),
        ("High_Volatility", create_volatile_market(0.02)),
        ("Low_Volatility", create_volatile_market(0.005)),
        ("Mean_Reverting", create_mean_reverting_market()),
    ];
    
    let strategies = vec![
        ("MeanReversion", create_mean_reversion_strategy()),
        ("TrendFollowing", create_trend_following_strategy()),
        ("MarketMaking", create_market_making_strategy()),
    ];
    
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission_per_contract: 0.5,
        ..Default::default()
    };
    
    // Results matrix: regime -> strategy -> performance
    let mut results_matrix: HashMap<String, HashMap<String, f64>> = HashMap::new();
    
    for (regime_name, market_data) in regimes {
        println!("Testing regime: {}", regime_name);
        let mut regime_results = HashMap::new();
        
        for (strategy_name, strategy) in strategies.clone() {
            let mut engine = BacktestEngine::new(config.clone());
            engine.add_strategy(strategy).expect("Failed to add strategy");
            
            let data_source = MockDataSource::new(market_data.clone());
            let report = engine.run_with_data_source(Box::new(data_source))
                .expect("Backtest failed");
            
            if let Some(result) = report.strategy_results.first() {
                regime_results.insert(strategy_name.to_string(), result.total_pnl);
            }
        }
        
        results_matrix.insert(regime_name.to_string(), regime_results);
    }
    
    // Display regime-specific results
    println!("\n=== Regime-Specific P&L Matrix ===\n");
    println!("{:<20} {:>15} {:>15} {:>15}",
             "Regime", "MeanReversion", "TrendFollowing", "MarketMaking");
    println!("{:-<65}", "");
    
    for regime in ["Trending_Up", "Trending_Down", "High_Volatility", "Low_Volatility", "Mean_Reverting"] {
        if let Some(results) = results_matrix.get(regime) {
            println!("{:<20} {:>15.2} {:>15.2} {:>15.2}",
                     regime,
                     results.get("MeanReversion").unwrap_or(&0.0),
                     results.get("TrendFollowing").unwrap_or(&0.0),
                     results.get("MarketMaking").unwrap_or(&0.0));
        }
    }
    
    // Identify best strategy for each regime
    println!("\n=== Best Strategy by Regime ===");
    for (regime, results) in &results_matrix {
        if let Some((best_strategy, best_pnl)) = results.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
            println!("{}: {} (P&L: ${:.2})", regime, best_strategy, best_pnl);
        }
    }
    
    println!("\nRegime-specific performance test completed!");
}

#[test]
fn test_parameter_sensitivity() {
    println!("=== Parameter Sensitivity Test ===\n");
    
    let market_data = create_multi_regime_market_data();
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission_per_contract: 0.5,
        ..Default::default()
    };
    
    // Test mean reversion with different lookback periods
    let lookback_periods = vec![10, 20, 50, 100, 200];
    let mut mr_results = Vec::new();
    
    println!("Testing Mean Reversion lookback sensitivity:");
    for period in lookback_periods {
        let strategy = MeanReversionStrategy::new(
            format!("MR_{}", period),
            1,
            MeanReversionConfig {
                lookback_period: period,
                entry_threshold: 1.5,
                exit_threshold: 0.5,
                ..Default::default()
            },
        );
        
        let mut engine = BacktestEngine::new(config.clone());
        engine.add_strategy(Box::new(strategy)).expect("Failed to add strategy");
        
        let data_source = MockDataSource::new(market_data.clone());
        let report = engine.run_with_data_source(Box::new(data_source))
            .expect("Backtest failed");
        
        if let Some(result) = report.strategy_results.first() {
            mr_results.push((period, result.total_pnl, report.performance_metrics.sharpe_ratio));
            println!("  Lookback {}: P&L=${:.2}, Sharpe={:.3}",
                     period, result.total_pnl, report.performance_metrics.sharpe_ratio);
        }
    }
    
    // Test trend following with different MA periods
    let ma_ratios = vec![(5, 20), (10, 30), (20, 50), (50, 200)];
    let mut tf_results = Vec::new();
    
    println!("\nTesting Trend Following MA period sensitivity:");
    for (fast, slow) in ma_ratios {
        let strategy = TrendFollowingStrategy::new(
            format!("TF_{}_{}", fast, slow),
            1,
            TrendFollowingConfig {
                fast_ma_period: fast,
                slow_ma_period: slow,
                atr_period: 14,
                ..Default::default()
            },
        );
        
        let mut engine = BacktestEngine::new(config.clone());
        engine.add_strategy(Box::new(strategy)).expect("Failed to add strategy");
        
        let data_source = MockDataSource::new(market_data.clone());
        let report = engine.run_with_data_source(Box::new(data_source))
            .expect("Backtest failed");
        
        if let Some(result) = report.strategy_results.first() {
            tf_results.push(((fast, slow), result.total_pnl, report.performance_metrics.sharpe_ratio));
            println!("  MA {}/{}: P&L=${:.2}, Sharpe={:.3}",
                     fast, slow, result.total_pnl, report.performance_metrics.sharpe_ratio);
        }
    }
    
    println!("\nParameter sensitivity test completed!");
}

#[test]
fn test_correlation_analysis() {
    println!("=== Strategy Correlation Analysis ===\n");
    
    // Run multiple strategies and collect their daily returns
    let market_data = create_multi_regime_market_data();
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission_per_contract: 0.5,
        ..Default::default()
    };
    
    let strategies = vec![
        ("MR1", create_mean_reversion_strategy()),
        ("MR2", create_alternative_mr_strategy()),
        ("TF1", create_trend_following_strategy()),
        ("MM1", create_market_making_strategy()),
    ];
    
    // Collect daily returns for each strategy
    let mut daily_returns: HashMap<String, Vec<f64>> = HashMap::new();
    
    for (name, strategy) in strategies {
        let mut engine = BacktestEngine::new(config.clone());
        engine.add_strategy(strategy).expect("Failed to add strategy");
        
        let data_source = MockDataSource::new(market_data.clone());
        let report = engine.run_with_data_source(Box::new(data_source))
            .expect("Backtest failed");
        
        // Extract daily returns from equity curve
        let mut returns = Vec::new();
        let equity_points = &report.equity_curve;
        
        for i in 1..equity_points.len() {
            let daily_return = (equity_points[i].equity - equity_points[i-1].equity) 
                              / equity_points[i-1].equity;
            returns.push(daily_return);
        }
        
        daily_returns.insert(name.to_string(), returns);
    }
    
    // Calculate correlations
    println!("Strategy Correlation Matrix:\n");
    let strategy_names: Vec<&String> = daily_returns.keys().collect();
    
    print!("{:<10}", "");
    for name in &strategy_names {
        print!("{:>10}", name);
    }
    println!();
    
    for name1 in &strategy_names {
        print!("{:<10}", name1);
        for name2 in &strategy_names {
            let returns1 = &daily_returns[*name1];
            let returns2 = &daily_returns[*name2];
            let correlation = calculate_correlation(returns1, returns2);
            print!("{:>10.3}", correlation);
        }
        println!();
    }
    
    println!("\nCorrelation analysis completed!");
}

// Helper functions

fn create_multi_regime_market_data() -> Vec<MarketEvent> {
    let mut events = Vec::new();
    let mut timestamp = 1_000_000;
    let mut price = 10000;
    
    // Trending up phase
    for i in 0..2000 {
        timestamp += 1000;
        price += ((rand_float() - 0.3) * 20.0) as i64;
        events.push(create_trade_event(1, price, timestamp, i));
        
        if i % 5 == 0 {
            events.push(create_bbo_event(1, price, timestamp + 100));
        }
    }
    
    // Volatile phase
    for i in 2000..4000 {
        timestamp += 1000;
        price += ((rand_float() - 0.5) * 50.0) as i64;
        events.push(create_trade_event(1, price, timestamp, i));
        
        if i % 5 == 0 {
            events.push(create_bbo_event(1, price, timestamp + 100));
        }
    }
    
    // Mean reverting phase
    let mean_price = price;
    for i in 4000..6000 {
        timestamp += 1000;
        let deviation = price - mean_price;
        price += ((-deviation as f64 * 0.1) + (rand_float() - 0.5) * 20.0) as i64;
        events.push(create_trade_event(1, price, timestamp, i));
        
        if i % 5 == 0 {
            events.push(create_bbo_event(1, price, timestamp + 100));
        }
    }
    
    events
}

fn create_trending_market(up: bool, strength: f64) -> Vec<MarketEvent> {
    let mut events = Vec::new();
    let mut timestamp = 1_000_000;
    let mut price = 10000;
    let bias = if up { strength } else { -strength };
    
    for i in 0..1000 {
        timestamp += 1000;
        price += ((rand_float() - 0.5 + bias) * 20.0) as i64;
        price = price.max(5000); // Floor price
        
        events.push(create_trade_event(1, price, timestamp, i));
        
        if i % 10 == 0 {
            events.push(create_bbo_event(1, price, timestamp + 100));
        }
    }
    
    events
}

fn create_volatile_market(volatility: f64) -> Vec<MarketEvent> {
    let mut events = Vec::new();
    let mut timestamp = 1_000_000;
    let mut price = 10000;
    
    for i in 0..1000 {
        timestamp += 1000;
        price += ((rand_float() - 0.5) * volatility * price as f64) as i64;
        price = price.max(5000); // Floor price
        
        events.push(create_trade_event(1, price, timestamp, i));
        
        if i % 10 == 0 {
            events.push(create_bbo_event(1, price, timestamp + 100));
        }
    }
    
    events
}

fn create_mean_reverting_market() -> Vec<MarketEvent> {
    let mut events = Vec::new();
    let mut timestamp = 1_000_000;
    let mut price = 10000;
    let mean = 10000;
    let mean_reversion_strength = 0.1;
    
    for i in 0..1000 {
        timestamp += 1000;
        let deviation = price - mean;
        price += ((-deviation as f64 * mean_reversion_strength) + 
                 (rand_float() - 0.5) * 30.0) as i64;
        
        events.push(create_trade_event(1, price, timestamp, i));
        
        if i % 10 == 0 {
            events.push(create_bbo_event(1, price, timestamp + 100));
        }
    }
    
    events
}

fn create_trade_event(instrument_id: InstrumentId, price: i64, timestamp: u64, id: u64) -> MarketEvent {
    MarketEvent::Trade(TradeEvent {
        instrument_id,
        trade_id: id,
        price: Price::new(price),
        quantity: Quantity::from((50 + (id % 100) as u32)),
        aggressor_side: if id % 2 == 0 { Side::Bid } else { Side::Ask },
        timestamp,
        buyer_order_id: None,
        seller_order_id: None,
    })
}

fn create_bbo_event(instrument_id: InstrumentId, mid_price: i64, timestamp: u64) -> MarketEvent {
    let spread = 25 + (rand_float() * 25.0) as i64;
    
    MarketEvent::BBO(BBOUpdate {
        instrument_id,
        bid_price: Some(Price::new(mid_price - spread / 2)),
        ask_price: Some(Price::new(mid_price + spread / 2)),
        bid_quantity: Some(Quantity::from((100 + (rand_float() * 200.0) as u32))),
        ask_quantity: Some(Quantity::from((100 + (rand_float() * 200.0) as u32))),
        timestamp,
    })
}

fn create_mean_reversion_strategy() -> Box<dyn Strategy> {
    Box::new(MeanReversionStrategy::new(
        "MeanReversion".to_string(),
        1,
        MeanReversionConfig {
            lookback_period: 20,
            entry_threshold: 1.5,
            exit_threshold: 0.5,
            ..Default::default()
        },
    ))
}

fn create_alternative_mr_strategy() -> Box<dyn Strategy> {
    Box::new(MeanReversionStrategy::new(
        "MeanReversionAlt".to_string(),
        1,
        MeanReversionConfig {
            lookback_period: 50,
            entry_threshold: 2.0,
            exit_threshold: 0.3,
            ..Default::default()
        },
    ))
}

fn create_trend_following_strategy() -> Box<dyn Strategy> {
    Box::new(TrendFollowingStrategy::new(
        "TrendFollowing".to_string(),
        1,
        TrendFollowingConfig {
            fast_ma_period: 10,
            slow_ma_period: 30,
            ..Default::default()
        },
    ))
}

fn create_market_making_strategy() -> Box<dyn Strategy> {
    Box::new(MarketMakerStrategy::new(
        "MarketMaking".to_string(),
        1,
        MarketMakerConfig::default(),
    ))
}

fn calculate_correlation(returns1: &[f64], returns2: &[f64]) -> f64 {
    if returns1.len() != returns2.len() || returns1.is_empty() {
        return 0.0;
    }
    
    let n = returns1.len() as f64;
    let mean1: f64 = returns1.iter().sum::<f64>() / n;
    let mean2: f64 = returns2.iter().sum::<f64>() / n;
    
    let mut cov = 0.0;
    let mut var1 = 0.0;
    let mut var2 = 0.0;
    
    for i in 0..returns1.len() {
        let diff1 = returns1[i] - mean1;
        let diff2 = returns2[i] - mean2;
        cov += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }
    
    if var1 == 0.0 || var2 == 0.0 {
        return 0.0;
    }
    
    cov / (var1.sqrt() * var2.sqrt())
}

// Simple random float generator
fn rand_float() -> f64 {
    use std::cell::RefCell;
    
    thread_local! {
        static STATE: RefCell<u64> = RefCell::new(42);
    }
    
    STATE.with(|state| {
        let mut s = state.borrow_mut();
        *s = s.wrapping_mul(1103515245).wrapping_add(12345);
        (*s as f64) / (u64::MAX as f64)
    })
}

/// Mock data source implementation
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
            let event = &self.events[self.index];
            self.index += 1;
            
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
                MarketEvent::BBO(bbo) => {
                    Some(algotrading::core::MarketUpdate::BBO(algotrading::core::BBO {
                        instrument_id: bbo.instrument_id,
                        bid_price: bbo.bid_price.unwrap_or(Price::new(0)),
                        ask_price: bbo.ask_price.unwrap_or(Price::new(0)),
                        bid_quantity: bbo.bid_quantity.unwrap_or(Quantity::from(0u32)),
                        ask_quantity: bbo.ask_quantity.unwrap_or(Quantity::from(0u32)),
                        timestamp: bbo.timestamp,
                    }))
                }
                _ => self.next_update(),
            }
        } else {
            None
        }
    }
}