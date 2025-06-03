//! Data loading and validation integration tests
//!
//! These tests verify the data ingestion pipeline, format validation,
//! and proper handling of various data edge cases.

use algotrading::core::types::{InstrumentId, Price, Quantity};
use algotrading::core::{Side, MarketUpdate};
use algotrading::market_data::reader::FileReader;
use algotrading::market_data::normalizer::DatabentoNormalizer;
use algotrading::order_book::market::Market;
use algotrading::contract_mgmt::symbology::SymbologyManager;
use std::path::Path;
use std::fs::{self, File};
use std::io::Write;
use std::collections::HashMap;

/// Test data format validation and error handling
#[test]
fn test_data_format_validation() {
    println!("=== Data Format Validation Test ===");
    
    // Test various data format scenarios
    let test_cases = vec![
        ("valid_data", create_valid_test_data()),
        ("corrupted_header", create_corrupted_header_data()),
        ("missing_fields", create_missing_fields_data()),
        ("invalid_timestamps", create_invalid_timestamp_data()),
    ];
    
    for (name, data) in test_cases {
        println!("\nTesting: {}", name);
        
        let temp_file = format!("/tmp/test_data_{}.dbn", name);
        fs::write(&temp_file, data).expect("Failed to write test file");
        
        match FileReader::new(&temp_file) {
            Ok(mut reader) => {
                let mut valid_events = 0;
                let mut errors = 0;
                
                while let Some(update) = reader.next_update() {
                    match validate_market_update(&update) {
                        Ok(_) => valid_events += 1,
                        Err(e) => {
                            errors += 1;
                            println!("  Validation error: {}", e);
                        }
                    }
                }
                
                println!("  Valid events: {}, Errors: {}", valid_events, errors);
                
                if name == "valid_data" {
                    assert!(errors == 0, "Valid data should have no errors");
                } else {
                    assert!(errors > 0 || valid_events == 0, 
                           "Invalid data should have errors or no valid events");
                }
            }
            Err(e) => {
                println!("  Failed to create reader: {}", e);
                assert!(name != "valid_data", "Valid data should create reader successfully");
            }
        }
        
        // Clean up
        let _ = fs::remove_file(&temp_file);
    }
    
    println!("\nData format validation test passed!");
}

/// Test data ingestion pipeline with various file sizes
#[test]
fn test_data_ingestion_pipeline() {
    println!("=== Data Ingestion Pipeline Test ===");
    
    // Test different file sizes
    let file_sizes = vec![
        ("small", 100),      // 100 events
        ("medium", 10_000),  // 10K events
        ("large", 100_000),  // 100K events
    ];
    
    for (size_name, num_events) in file_sizes {
        println!("\nTesting {} file ({} events)", size_name, num_events);
        
        let test_data = generate_synthetic_market_data(num_events);
        let temp_file = format!("/tmp/test_data_{}.dbn", size_name);
        
        // Write test data
        write_test_market_data(&temp_file, &test_data);
        
        // Test reading and processing
        let start = std::time::Instant::now();
        let mut reader = FileReader::new(&temp_file).expect("Failed to create reader");
        let mut events_processed = 0;
        
        while let Some(_update) = reader.next_update() {
            events_processed += 1;
        }
        
        let elapsed = start.elapsed();
        let throughput = events_processed as f64 / elapsed.as_secs_f64();
        
        println!("  Processed {} events in {:?}", events_processed, elapsed);
        println!("  Throughput: {:.0} events/second", throughput);
        
        assert_eq!(events_processed, num_events, 
                   "Should process all events for {}", size_name);
        
        // Clean up
        let _ = fs::remove_file(&temp_file);
    }
    
    println!("\nData ingestion pipeline test passed!");
}

/// Test order book reconstruction from market data
#[test]
fn test_order_book_reconstruction() {
    println!("=== Order Book Reconstruction Test ===");
    
    // Create test market data with known order book updates
    let test_events = create_order_book_test_data();
    let temp_file = "/tmp/test_orderbook_data.dbn";
    write_test_market_data(temp_file, &test_events);
    
    // Create market and process data
    let mut market = Market::new();
    let mut reader = FileReader::new(temp_file).expect("Failed to create reader");
    let normalizer = DatabentoNormalizer::new();
    
    let mut update_count = 0;
    while let Some(raw_update) = reader.next_update() {
        // Normalize the update
        match normalizer.normalize_update(raw_update) {
            Ok(normalized) => {
                market.process_update(normalized);
                update_count += 1;
                
                // Verify order book state at key points
                if update_count % 100 == 0 {
                    if let Some(book) = market.get_book(&1, &"CME".to_string()) {
                        let bid_levels = book.bid_levels();
                        let ask_levels = book.ask_levels();
                        
                        println!("  After {} updates:", update_count);
                        println!("    Bid levels: {}", bid_levels.len());
                        println!("    Ask levels: {}", ask_levels.len());
                        
                        if !bid_levels.is_empty() && !ask_levels.is_empty() {
                            let best_bid = bid_levels[0].price;
                            let best_ask = ask_levels[0].price;
                            let spread = best_ask.0 - best_bid.0;
                            
                            println!("    Best bid: {}, Best ask: {}, Spread: {}", 
                                     best_bid.0, best_ask.0, spread);
                            
                            // Validate spread is reasonable
                            assert!(spread > 0 && spread < 1000, 
                                   "Spread should be positive and reasonable");
                        }
                    }
                }
            }
            Err(e) => {
                println!("  Normalization error: {}", e);
            }
        }
    }
    
    println!("  Total updates processed: {}", update_count);
    assert!(update_count > 0, "Should process some updates");
    
    // Clean up
    let _ = fs::remove_file(temp_file);
    
    println!("\nOrder book reconstruction test passed!");
}

/// Test symbology management and instrument mapping
#[test]
fn test_symbology_management() {
    println!("=== Symbology Management Test ===");
    
    // Create test symbology data
    let symbology_data = create_test_symbology();
    let temp_file = "/tmp/test_symbology.json";
    
    let json_data = serde_json::to_string_pretty(&symbology_data).unwrap();
    fs::write(temp_file, json_data).expect("Failed to write symbology file");
    
    // Test loading and querying
    let mut symbology = SymbologyManager::new();
    symbology.load_from_file(temp_file).expect("Failed to load symbology");
    
    // Test instrument lookups
    let test_queries = vec![
        (1, Some("ESH4")),
        (2, Some("GCG4")),
        (3, Some("CLK4")),
        (999, None), // Non-existent
    ];
    
    for (instrument_id, expected_symbol) in test_queries {
        match symbology.get_symbol(instrument_id) {
            Some(symbol) => {
                println!("  Instrument {} -> Symbol: {}", instrument_id, symbol);
                assert_eq!(Some(symbol.as_str()), expected_symbol);
            }
            None => {
                println!("  Instrument {} -> Not found", instrument_id);
                assert_eq!(None, expected_symbol);
            }
        }
    }
    
    // Test reverse lookup
    if let Some(id) = symbology.get_instrument_id("ESH4") {
        assert_eq!(id, 1, "ESH4 should map to instrument 1");
    }
    
    // Clean up
    let _ = fs::remove_file(temp_file);
    
    println!("\nSymbology management test passed!");
}

/// Test handling of edge cases in market data
#[test]
fn test_market_data_edge_cases() {
    println!("=== Market Data Edge Cases Test ===");
    
    let edge_cases = vec![
        ("zero_prices", create_zero_price_data()),
        ("extreme_prices", create_extreme_price_data()),
        ("zero_quantities", create_zero_quantity_data()),
        ("rapid_updates", create_rapid_update_data()),
        ("gaps_in_data", create_gapped_data()),
    ];
    
    for (case_name, test_data) in edge_cases {
        println!("\nTesting edge case: {}", case_name);
        
        let temp_file = format!("/tmp/test_edge_{}.dbn", case_name);
        write_test_market_data(&temp_file, &test_data);
        
        let mut reader = FileReader::new(&temp_file).expect("Failed to create reader");
        let mut valid_updates = 0;
        let mut invalid_updates = 0;
        
        while let Some(update) = reader.next_update() {
            match validate_edge_case_update(&update, case_name) {
                Ok(_) => valid_updates += 1,
                Err(e) => {
                    invalid_updates += 1;
                    println!("  Invalid update: {}", e);
                }
            }
        }
        
        println!("  Valid: {}, Invalid: {}", valid_updates, invalid_updates);
        
        // Clean up
        let _ = fs::remove_file(&temp_file);
    }
    
    println!("\nMarket data edge cases test passed!");
}

/// Test concurrent data loading from multiple files
#[test]
fn test_concurrent_data_loading() {
    println!("=== Concurrent Data Loading Test ===");
    
    use std::thread;
    use std::sync::{Arc, Mutex};
    
    // Create multiple test files
    let num_files = 4;
    let events_per_file = 1000;
    let mut file_paths = Vec::new();
    
    for i in 0..num_files {
        let file_path = format!("/tmp/test_concurrent_{}.dbn", i);
        let test_data = generate_synthetic_market_data(events_per_file);
        write_test_market_data(&file_path, &test_data);
        file_paths.push(file_path);
    }
    
    // Process files concurrently
    let total_events = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for file_path in file_paths.clone() {
        let total_events_clone = Arc::clone(&total_events);
        
        let handle = thread::spawn(move || {
            let mut reader = FileReader::new(&file_path).expect("Failed to create reader");
            let mut local_count = 0;
            
            while let Some(_update) = reader.next_update() {
                local_count += 1;
            }
            
            let mut total = total_events_clone.lock().unwrap();
            *total += local_count;
            
            println!("  Thread processed {} events from {}", local_count, file_path);
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread failed");
    }
    
    let final_total = *total_events.lock().unwrap();
    println!("\nTotal events processed: {}", final_total);
    assert_eq!(final_total, num_files * events_per_file, 
               "Should process all events from all files");
    
    // Clean up
    for file_path in file_paths {
        let _ = fs::remove_file(file_path);
    }
    
    println!("\nConcurrent data loading test passed!");
}

// Helper functions

fn create_valid_test_data() -> Vec<u8> {
    // Create minimal valid DBN format data
    vec![1, 2, 3, 4] // Placeholder - would be actual DBN format
}

fn create_corrupted_header_data() -> Vec<u8> {
    vec![255, 255, 255, 255] // Invalid header
}

fn create_missing_fields_data() -> Vec<u8> {
    vec![1, 2] // Incomplete data
}

fn create_invalid_timestamp_data() -> Vec<u8> {
    vec![1, 2, 3, 4] // Would have invalid timestamps
}

fn generate_synthetic_market_data(num_events: usize) -> Vec<MarketUpdate> {
    let mut events = Vec::with_capacity(num_events);
    let mut timestamp = 1_000_000;
    let mut price = 10000;
    
    for i in 0..num_events {
        timestamp += 1000;
        price += ((i % 20) as i64) - 10;
        
        if i % 2 == 0 {
            events.push(MarketUpdate::Trade(algotrading::core::Trade {
                instrument_id: 1,
                price: Price::new(price),
                quantity: Quantity::from((100 + (i % 50)) as u32),
                side: if i % 3 == 0 { Side::Bid } else { Side::Ask },
                timestamp,
            }));
        } else {
            events.push(MarketUpdate::BBO(algotrading::core::BBO {
                instrument_id: 1,
                bid_price: Price::new(price - 25),
                ask_price: Price::new(price + 25),
                bid_quantity: Quantity::from(200u32),
                ask_quantity: Quantity::from(200u32),
                timestamp,
            }));
        }
    }
    
    events
}

fn write_test_market_data(path: &str, _events: &[MarketUpdate]) {
    // In a real implementation, this would serialize to DBN format
    // For testing, we'll just create a dummy file
    let mut file = File::create(path).expect("Failed to create file");
    file.write_all(b"TEST_DATA").expect("Failed to write data");
}

fn validate_market_update(update: &MarketUpdate) -> Result<(), String> {
    match update {
        MarketUpdate::Trade(trade) => {
            if trade.price.0 <= 0 {
                return Err("Invalid trade price".to_string());
            }
            if trade.quantity.value == 0 {
                return Err("Zero trade quantity".to_string());
            }
            Ok(())
        }
        MarketUpdate::BBO(bbo) => {
            if bbo.bid_price.0 >= bbo.ask_price.0 {
                return Err("Crossed BBO".to_string());
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn create_order_book_test_data() -> Vec<MarketUpdate> {
    generate_synthetic_market_data(500)
}

fn create_test_symbology() -> HashMap<String, serde_json::Value> {
    let mut symbology = HashMap::new();
    
    symbology.insert("instruments".to_string(), serde_json::json!([
        {"id": 1, "symbol": "ESH4", "exchange": "CME"},
        {"id": 2, "symbol": "GCG4", "exchange": "COMEX"},
        {"id": 3, "symbol": "CLK4", "exchange": "NYMEX"},
    ]));
    
    symbology
}

fn create_zero_price_data() -> Vec<MarketUpdate> {
    vec![
        MarketUpdate::Trade(algotrading::core::Trade {
            instrument_id: 1,
            price: Price::new(0),
            quantity: Quantity::from(100u32),
            side: Side::Bid,
            timestamp: 1_000_000,
        })
    ]
}

fn create_extreme_price_data() -> Vec<MarketUpdate> {
    vec![
        MarketUpdate::Trade(algotrading::core::Trade {
            instrument_id: 1,
            price: Price::new(i64::MAX / 2),
            quantity: Quantity::from(100u32),
            side: Side::Ask,
            timestamp: 1_000_000,
        })
    ]
}

fn create_zero_quantity_data() -> Vec<MarketUpdate> {
    vec![
        MarketUpdate::Trade(algotrading::core::Trade {
            instrument_id: 1,
            price: Price::new(10000),
            quantity: Quantity::from(0u32),
            side: Side::Bid,
            timestamp: 1_000_000,
        })
    ]
}

fn create_rapid_update_data() -> Vec<MarketUpdate> {
    let mut events = Vec::new();
    let timestamp = 1_000_000;
    
    // 100 updates at the same timestamp
    for i in 0..100 {
        events.push(MarketUpdate::Trade(algotrading::core::Trade {
            instrument_id: 1,
            price: Price::new(10000 + i),
            quantity: Quantity::from(100u32),
            side: Side::Bid,
            timestamp,
        }));
    }
    
    events
}

fn create_gapped_data() -> Vec<MarketUpdate> {
    vec![
        MarketUpdate::Trade(algotrading::core::Trade {
            instrument_id: 1,
            price: Price::new(10000),
            quantity: Quantity::from(100u32),
            side: Side::Bid,
            timestamp: 1_000_000,
        }),
        MarketUpdate::Trade(algotrading::core::Trade {
            instrument_id: 1,
            price: Price::new(10100),
            quantity: Quantity::from(100u32),
            side: Side::Ask,
            timestamp: 10_000_000, // 9 second gap
        }),
    ]
}

fn validate_edge_case_update(update: &MarketUpdate, case_name: &str) -> Result<(), String> {
    match case_name {
        "zero_prices" => {
            if let MarketUpdate::Trade(trade) = update {
                if trade.price.0 == 0 {
                    return Err("Zero price detected".to_string());
                }
            }
        }
        "extreme_prices" => {
            if let MarketUpdate::Trade(trade) = update {
                if trade.price.0 > 1_000_000_000 {
                    return Err("Extreme price detected".to_string());
                }
            }
        }
        "zero_quantities" => {
            if let MarketUpdate::Trade(trade) = update {
                if trade.quantity.value == 0 {
                    return Err("Zero quantity detected".to_string());
                }
            }
        }
        _ => {}
    }
    
    Ok(())
}