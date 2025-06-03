//! Performance benchmarks for the backtesting system
//!
//! These benchmarks measure throughput and latency of various components
//! to ensure the system meets the 18M messages/second target.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use algotrading::core::types::{InstrumentId, Price, Quantity};
use algotrading::core::Side;
use algotrading::market_data::events::{MarketEvent, TradeEvent, BBOUpdate};
use algotrading::strategy::{Strategy, StrategyConfig, StrategyContext, StrategyOutput};
use algotrading::features::{FeatureExtractor, FeatureConfig};
use algotrading::backtest::engine::{BacktestEngine, BacktestConfig};
use algotrading::backtest::execution::{LatencyModel, FillModel};
use algotrading::strategies::{MeanReversionStrategy, mean_reversion::MeanReversionConfig};
use algotrading::strategies::{MarketMakerStrategy, market_maker::MarketMakerConfig};
use algotrading::strategies::{TrendFollowingStrategy, trend_following::TrendFollowingConfig};
use std::time::Duration;

/// Generate a batch of market events for benchmarking
fn generate_benchmark_events(num_events: usize) -> Vec<MarketEvent> {
    let mut events = Vec::with_capacity(num_events);
    let mut timestamp = 1_000_000;
    let mut price = 10000;
    
    for i in 0..num_events {
        timestamp += 100; // 100 microseconds between events
        
        if i % 2 == 0 {
            // Trade event
            events.push(MarketEvent::Trade(TradeEvent {
                instrument_id: 1,
                trade_id: i as u64,
                price: Price::new(price + ((i % 10) as i64 - 5)),
                quantity: Quantity::from((100 + (i % 50)) as u32),
                aggressor_side: if i % 3 == 0 { Side::Bid } else { Side::Ask },
                timestamp,
                buyer_order_id: None,
                seller_order_id: None,
            }));
        } else {
            // BBO event
            let spread = 25 + (i % 25) as i64;
            events.push(MarketEvent::BBO(BBOUpdate {
                instrument_id: 1,
                bid_price: Some(Price::new(price - spread / 2)),
                ask_price: Some(Price::new(price + spread / 2)),
                bid_quantity: Some(Quantity::from((200 + (i % 100)) as u32)),
                ask_quantity: Some(Quantity::from((200 + (i % 100)) as u32)),
                timestamp,
            }));
        }
    }
    
    events
}

/// Mock data source for benchmarking
struct BenchmarkDataSource {
    events: Vec<MarketEvent>,
    index: usize,
}

impl BenchmarkDataSource {
    fn new(events: Vec<MarketEvent>) -> Self {
        Self { events, index: 0 }
    }
}

impl algotrading::core::traits::MarketDataSource for BenchmarkDataSource {
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

fn benchmark_event_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_processing");
    
    for num_events in [1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_events),
            num_events,
            |b, &num_events| {
                let events = generate_benchmark_events(num_events);
                
                b.iter(|| {
                    let config = BacktestConfig {
                        initial_capital: 100_000.0,
                        calculate_features: false,
                        ..Default::default()
                    };
                    
                    let mut engine = BacktestEngine::new(config);
                    let data_source = BenchmarkDataSource::new(events.clone());
                    
                    let _report = engine.run_with_data_source(Box::new(data_source))
                        .expect("Backtest failed");
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_strategy_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_execution");
    
    let strategies: Vec<(&str, Box<dyn Strategy>)> = vec![
        ("mean_reversion", Box::new(MeanReversionStrategy::new(
            "MeanReversion".to_string(),
            1,
            MeanReversionConfig::default(),
        ))),
        ("market_maker", Box::new(MarketMakerStrategy::new(
            "MarketMaker".to_string(),
            1,
            MarketMakerConfig::default(),
        ))),
        ("trend_following", Box::new(TrendFollowingStrategy::new(
            "TrendFollowing".to_string(),
            1,
            TrendFollowingConfig::default(),
        ))),
    ];
    
    for (name, strategy) in strategies {
        group.bench_function(name, |b| {
            let events = generate_benchmark_events(10_000);
            
            b.iter(|| {
                let config = BacktestConfig {
                    initial_capital: 100_000.0,
                    ..Default::default()
                };
                
                let mut engine = BacktestEngine::new(config);
                engine.add_strategy(strategy.clone()).expect("Failed to add strategy");
                
                let data_source = BenchmarkDataSource::new(events.clone());
                let _report = engine.run_with_data_source(Box::new(data_source))
                    .expect("Backtest failed");
            });
        });
    }
    
    group.finish();
}

fn benchmark_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");
    
    let feature_config = FeatureConfig {
        book_features_enabled: true,
        flow_features_enabled: true,
        rolling_features_enabled: true,
        book_levels: 5,
        flow_window_size: 100,
        rolling_windows: vec![10, 30, 60],
    };
    
    group.bench_function("feature_calculation", |b| {
        let events = generate_benchmark_events(1000);
        
        b.iter(|| {
            let config = BacktestConfig {
                initial_capital: 100_000.0,
                calculate_features: true,
                feature_config: feature_config.clone(),
                ..Default::default()
            };
            
            let mut engine = BacktestEngine::new(config);
            let data_source = BenchmarkDataSource::new(events.clone());
            
            let _report = engine.run_with_data_source(Box::new(data_source))
                .expect("Backtest failed");
        });
    });
    
    group.finish();
}

fn benchmark_order_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_execution");
    
    let latency_models = vec![
        ("fixed", LatencyModel::Fixed(100)),
        ("variable", LatencyModel::Variable { min: 50, max: 200 }),
        ("size_dependent", LatencyModel::SizeDependent { 
            base: 100, 
            size_factor: 0.1 
        }),
    ];
    
    let fill_models = vec![
        ("optimistic", FillModel::Optimistic),
        ("midpoint", FillModel::Midpoint),
        ("realistic", FillModel::Realistic {
            maker_fill_prob: 0.7,
            taker_slippage_ticks: 1,
        }),
    ];
    
    for (latency_name, latency_model) in &latency_models {
        for (fill_name, fill_model) in &fill_models {
            let name = format!("{}_latency_{}_fill", latency_name, fill_name);
            
            group.bench_function(&name, |b| {
                let events = generate_benchmark_events(5000);
                
                b.iter(|| {
                    let config = BacktestConfig {
                        initial_capital: 100_000.0,
                        latency_model: latency_model.clone(),
                        fill_model: fill_model.clone(),
                        ..Default::default()
                    };
                    
                    let mut engine = BacktestEngine::new(config);
                    
                    // Add a simple strategy that generates orders
                    let strategy = SimpleOrderStrategy::new();
                    engine.add_strategy(Box::new(strategy)).expect("Failed to add strategy");
                    
                    let data_source = BenchmarkDataSource::new(events.clone());
                    let _report = engine.run_with_data_source(Box::new(data_source))
                        .expect("Backtest failed");
                });
            });
        }
    }
    
    group.finish();
}

/// Simple strategy that generates orders for benchmarking
struct SimpleOrderStrategy {
    config: StrategyConfig,
    order_count: usize,
}

impl SimpleOrderStrategy {
    fn new() -> Self {
        let config = StrategyConfig::new("SimpleOrder".to_string(), "Simple Order Strategy")
            .with_instrument(1)
            .with_max_position(10);
        
        Self { config, order_count: 0 }
    }
}

impl Strategy for SimpleOrderStrategy {
    fn initialize(&mut self, _context: &StrategyContext) -> Result<(), algotrading::strategy::StrategyError> {
        Ok(())
    }
    
    fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput {
        let mut output = StrategyOutput::default();
        
        // Generate an order every 10 events
        if self.order_count % 10 == 0 && context.position.quantity < 5 {
            if let MarketEvent::Trade(trade) = event {
                let order = algotrading::strategy::output::OrderRequest::market_order(
                    context.strategy_id.clone(),
                    trade.instrument_id,
                    algotrading::strategy::OrderSide::Buy,
                    Quantity::from(1u32),
                );
                output.orders.push(order);
            }
        }
        
        self.order_count += 1;
        output
    }
    
    fn config(&self) -> &StrategyConfig {
        &self.config
    }
}

fn benchmark_throughput_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(30)); // Longer measurement for accurate throughput
    
    group.bench_function("max_throughput", |b| {
        // Generate 1M events for throughput testing
        let events = generate_benchmark_events(1_000_000);
        
        b.iter(|| {
            let config = BacktestConfig {
                initial_capital: 100_000.0,
                calculate_features: false, // Disable features for max throughput
                ..Default::default()
            };
            
            let mut engine = BacktestEngine::new(config);
            
            // Add multiple strategies to simulate realistic load
            for i in 0..3 {
                let strategy = MeanReversionStrategy::new(
                    format!("MeanReversion_{}", i),
                    1,
                    MeanReversionConfig {
                        lookback_period: 20,
                        ..Default::default()
                    },
                );
                engine.add_strategy(Box::new(strategy)).expect("Failed to add strategy");
            }
            
            let data_source = BenchmarkDataSource::new(events.clone());
            let report = engine.run_with_data_source(Box::new(data_source))
                .expect("Backtest failed");
            
            black_box(report.events_processed);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_event_processing,
    benchmark_strategy_execution,
    benchmark_feature_extraction,
    benchmark_order_execution,
    benchmark_throughput_test
);
criterion_main!(benches);