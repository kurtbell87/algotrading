//! Simple performance test to measure zero-copy optimizations
//!
//! Tests the actual impact of our optimizations against the 18M target

use std::time::Instant;
use algotrading::core::types::{InstrumentId, Price, Quantity, Side, MarketUpdate, Trade};
use algotrading::market_data::{FastMarketState, FastFeatureVector, LAST_PRICE_IDX, VOLUME_IDX};
use std::collections::HashMap;

fn main() {
    println!("=== PERFORMANCE OPTIMIZATION TEST ===");
    println!("Goal: Achieve >15M events/s (>83% of 18M target)");
    println!("Previous Baseline: 3.1M events/s (17% of target)");
    
    // Test different event counts
    for &num_events in &[100_000, 500_000, 1_000_000] {
        println!("\n--- Testing with {} events ---", num_events);
        
        let events = generate_test_events(num_events);
        
        // Test 1: Raw event iteration (baseline)
        let raw_throughput = test_raw_iteration(&events);
        println!("1. Raw iteration: {:.0} events/s", raw_throughput);
        
        // Test 2: Event cloning overhead (current bottleneck) 
        let clone_throughput = test_event_cloning(&events);
        println!("2. With cloning: {:.0} events/s ({:.1}x slower than raw)", 
                clone_throughput, raw_throughput / clone_throughput);
        
        // Test 3: Optimized market state (hashbrown, lockless)
        let market_throughput = test_optimized_market_state(&events);
        println!("3. Optimized market: {:.0} events/s", market_throughput);
        
        // Test 4: Optimized features (array indexing)
        let feature_throughput = test_optimized_features(&events);
        println!("4. Optimized features: {:.0} events/s", feature_throughput);
        
        // Test 5: Combined optimizations (target test)
        let combined_throughput = test_combined_optimizations(&events);
        let efficiency = (combined_throughput / 18_000_000.0) * 100.0;
        
        println!("5. Combined optimized: {:.0} events/s ({:.1}% of 18M target)", 
                combined_throughput, efficiency);
        
        let improvement = combined_throughput / 3_100_000.0;
        println!("   Improvement vs baseline: {:.1}x", improvement);
        
        // Performance assessment
        if combined_throughput >= 15_000_000.0 {
            println!("🎯 SUCCESS: Achieved target performance!");
        } else if combined_throughput >= 10_000_000.0 {
            println!("✅ GOOD: Significant improvement");
        } else if combined_throughput >= 6_000_000.0 {
            println!("⚠️  MODERATE: Some improvement");
        } else {
            println!("❌ NEEDS WORK: Still below 6M threshold");
        }
    }
}

fn generate_test_events(count: usize) -> Vec<MarketUpdate> {
    let mut events = Vec::with_capacity(count);
    let mut price = 100_000_000i64;
    let mut timestamp = 1_000_000u64;
    
    for i in 0..count {
        timestamp += 100;
        price += ((i % 20) as i64) - 10;
        
        events.push(MarketUpdate::Trade(Trade {
            instrument_id: (i % 10 + 1) as InstrumentId,
            price: Price::new(price),
            quantity: Quantity::from((100 + (i % 100)) as u32),
            side: if i % 2 == 0 { Side::Bid } else { Side::Ask },
            timestamp,
        }));
    }
    
    events
}

fn test_raw_iteration(events: &[MarketUpdate]) -> f64 {
    let start = Instant::now();
    
    for event in events {
        std::hint::black_box(event);
    }
    
    let elapsed = start.elapsed();
    events.len() as f64 / elapsed.as_secs_f64()
}

fn test_event_cloning(events: &[MarketUpdate]) -> f64 {
    let start = Instant::now();
    let mut cloned_events = Vec::with_capacity(events.len());
    
    for event in events {
        cloned_events.push(event.clone()); // Simulate current bottleneck
        std::hint::black_box(&cloned_events.last());
    }
    
    let elapsed = start.elapsed();
    events.len() as f64 / elapsed.as_secs_f64()
}

fn test_optimized_market_state(events: &[MarketUpdate]) -> f64 {
    let start = Instant::now();
    let mut market_state = FastMarketState::new();
    
    for event in events {
        market_state.update(event);
        std::hint::black_box(market_state.get_last_price(1));
    }
    
    let elapsed = start.elapsed();
    events.len() as f64 / elapsed.as_secs_f64()
}

fn test_optimized_features(events: &[MarketUpdate]) -> f64 {
    let start = Instant::now();
    let mut features = FastFeatureVector::new(0);
    
    for event in events {
        if let MarketUpdate::Trade(trade) = event {
            features.update_from_trade(trade, None);
            std::hint::black_box(features.get(LAST_PRICE_IDX));
            std::hint::black_box(features.get(VOLUME_IDX));
        }
    }
    
    let elapsed = start.elapsed();
    events.len() as f64 / elapsed.as_secs_f64()
}

fn test_combined_optimizations(events: &[MarketUpdate]) -> f64 {
    let start = Instant::now();
    
    // Pre-allocate optimized structures
    let mut market_state = FastMarketState::new();
    let mut features = FastFeatureVector::new(0);
    let mut decision_count = 0u64;
    
    // Process in batches for cache efficiency
    const BATCH_SIZE: usize = 1000;
    for chunk in events.chunks(BATCH_SIZE) {
        for event in chunk {
            // Fast market state update (hashbrown HashMap, O(1))
            market_state.update(event);
            
            // Fast feature extraction (array indexing)
            if let MarketUpdate::Trade(trade) = event {
                let prev_price = market_state.get_last_price(trade.instrument_id);
                features.update_from_trade(trade, prev_price);
                
                // Simulate minimal strategy decision
                let price = features.get(LAST_PRICE_IDX);
                let volume = features.get(VOLUME_IDX);
                if price > 100.0 && volume > 500.0 {
                    decision_count += 1;
                }
            }
        }
    }
    
    let elapsed = start.elapsed();
    std::hint::black_box(decision_count);
    events.len() as f64 / elapsed.as_secs_f64()
}

// Traditional (slow) comparison tests for reference
#[allow(dead_code)]
fn test_traditional_market_state(events: &[MarketUpdate]) -> f64 {
    let start = Instant::now();
    let mut prices: HashMap<InstrumentId, Price> = HashMap::new();
    let mut timestamps: HashMap<InstrumentId, u64> = HashMap::new();
    
    for event in events {
        match event {
            MarketUpdate::Trade(trade) => {
                prices.insert(trade.instrument_id, trade.price);
                timestamps.insert(trade.instrument_id, trade.timestamp);
                std::hint::black_box(prices.get(&1));
            }
            _ => {}
        }
    }
    
    let elapsed = start.elapsed();
    events.len() as f64 / elapsed.as_secs_f64()
}

#[allow(dead_code)]
fn test_traditional_features(events: &[MarketUpdate]) -> f64 {
    let start = Instant::now();
    let mut features: HashMap<String, f64> = HashMap::new();
    
    for event in events {
        if let MarketUpdate::Trade(trade) = event {
            features.insert("last_price".to_string(), trade.price.as_f64());
            features.insert("volume".to_string(), trade.quantity.as_f64());
            std::hint::black_box(features.get("last_price"));
            std::hint::black_box(features.get("volume"));
        }
    }
    
    let elapsed = start.elapsed();
    events.len() as f64 / elapsed.as_secs_f64()
}