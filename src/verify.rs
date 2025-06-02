use algotrading::order_book::Market;
use databento::{
    dbn::{
        Dataset, Schema,
        decode::{DbnDecoder, DbnMetadata, DecodeRecord},
        record::MboMsg,
    },
    historical::timeseries::GetRangeToFileParams,
    HistoricalClient,
};
use std::{
    fs::File,
    io::BufReader,
    collections::HashMap,
};
use time::macros::datetime;
use memmap2::Mmap;

pub async fn run_verify_mode() -> Result<(), Box<dyn std::error::Error>> {
    // This matches Databento's LOB example exactly for verification
    // https://databento.com/docs/examples/order-book/limit-order-book/results
    
    let mut client = HistoricalClient::builder()
        .key_from_env()?
        .build()?;

    let data_path = "verify_lob_data.dbn.zst";
    println!("Downloading GOOG/GOOGL data for LOB verification...");

    // Using the exact same parameters as Databento's example
    client
        .timeseries()
        .get_range_to_file(
            &GetRangeToFileParams::builder()
                .dataset(Dataset::DbeqBasic)  // Databento Equities Basic
                .date_time_range((
                    datetime!(2024-04-03 08:00:00 UTC),
                    datetime!(2024-04-03 14:00:00 UTC),
                ))
                .symbols(vec!["GOOG", "GOOGL"])
                .schema(Schema::Mbo)
                .path(&data_path)
                .build(),
        )
        .await?;

    println!("Processing LOB data...");
    println!("\nExpected results from Databento's Python example:");
    println!("GOOGL: First BBO should transition from None/152.80 to 155.20/152.80");
    println!("Price levels should be around 152.80 (ask) and 155.20 (bid)");
    println!("\nActual results from our implementation:");
    
    // Process with direct MBO handling to match Databento's example
    process_verify_data(&data_path)?;

    Ok(())
}

fn process_verify_data(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let reader = BufReader::new(std::io::Cursor::new(&mmap[..]));
    let mut decoder = DbnDecoder::with_zstd_buffer(reader)?;
    
    // Get metadata to access symbol mappings
    let metadata = decoder.metadata();
    println!("\nMetadata version: {}", metadata.version);
    println!("Dataset: {}", metadata.dataset);
    println!("Schema: {:?}", metadata.schema);
    println!("Symbols: {:?}", metadata.symbols);
    
    // Create symbol mapping from metadata
    let mut symbol_map: HashMap<u32, String> = HashMap::new();
    let mappings = &metadata.mappings;
    println!("\nFound {} symbol mappings", mappings.len());
    for mapping in mappings {
        println!("Mapping: {} -> instrument_id {}", 
            mapping.raw_symbol, 
            mapping.intervals[0].symbol
        );
        // Extract instrument ID from the interval symbol field
        if let Ok(inst_id) = mapping.intervals[0].symbol.parse::<u32>() {
            symbol_map.insert(inst_id, mapping.raw_symbol.clone());
        }
    }
    
    let mut market = Market::new();
    let mut message_count = 0;
    
    // Count of messages printed for limiting output
    let mut printed_count = 0;
    const MAX_PRINTS: usize = 5;
    
    while let Some(mbo) = decoder.decode_record::<MboMsg>()? {
        message_count += 1;
        
        // Apply the message to the market
        market.apply(mbo.clone());
        
        // Print aggregated book state when F_LAST flag is set (like Databento's example)
        if mbo.flags.is_last() && printed_count < MAX_PRINTS {
            // Get aggregated BBO
            let (bid, ask) = market.aggregated_bbo(mbo.hd.instrument_id);
            
            // Get symbol from map
            let symbol = symbol_map.get(&mbo.hd.instrument_id)
                .map(|s| s.as_str())
                .unwrap_or("UNKNOWN");
            
            println!("{} Aggregated BBO | {}:",
                symbol,
                format_timestamp_with_offset(mbo.ts_recv)
            );
            
            match ask {
                Some(a) => println!("     {:>3} @ {:.9} | {:2} order(s)", 
                    a.size, a.price as f64 / 1e9, a.count),
                None => println!("    None"),
            }
            
            match bid {
                Some(b) => println!("     {:>3} @ {:.9} | {:2} order(s)", 
                    b.size, b.price as f64 / 1e9, b.count),
                None => println!("    None"),
            }
            
            printed_count += 1;
        }
    }
    
    println!("\nProcessed {} messages", message_count);
    
    // Print final state for all instruments
    println!("\nFinal order book state:");
    for inst_id in market.instruments() {
        let (bid, ask) = market.aggregated_bbo(inst_id);
        
        print!("Instrument {}: ", inst_id);
        match (bid, ask) {
            (Some(b), Some(a)) => println!("Bid: {} @ {:.2} | Ask: {} @ {:.2}", 
                b.size, b.price as f64 / 1e9, a.size, a.price as f64 / 1e9),
            (Some(b), None) => println!("Bid: {} @ {:.2} | Ask: None", 
                b.size, b.price as f64 / 1e9),
            (None, Some(a)) => println!("Bid: None | Ask: {} @ {:.2}", 
                a.size, a.price as f64 / 1e9),
            (None, None) => println!("No quotes"),
        }
    }
    
    Ok(())
}

#[allow(dead_code)]
fn format_timestamp(ts_ns: u64) -> String {
    let seconds = (ts_ns / 1_000_000_000) as i64;
    let nanos = (ts_ns % 1_000_000_000) as u32;
    
    // Create OffsetDateTime from Unix timestamp
    let dt = time::OffsetDateTime::from_unix_timestamp(seconds)
        .expect("valid timestamp");
    
    // Format as ISO 8601 with nanoseconds
    format!("{}-{:02}-{:02} {:02}:{:02}:{:02}.{:09}",
        dt.year(), dt.month() as u8, dt.day(),
        dt.hour(), dt.minute(), dt.second(), nanos)
}

fn format_timestamp_with_offset(ts_ns: u64) -> String {
    let seconds = (ts_ns / 1_000_000_000) as i64;
    let nanos = (ts_ns % 1_000_000_000) as u32;
    
    // Create OffsetDateTime from Unix timestamp
    let dt = time::OffsetDateTime::from_unix_timestamp(seconds)
        .expect("valid timestamp");
    
    // Format as ISO 8601 with nanoseconds and timezone offset
    format!("{}-{:02}-{:02} {:02}:{:02}:{:02}.{:09}+00:00",
        dt.year(), dt.month() as u8, dt.day(),
        dt.hour(), dt.minute(), dt.second(), nanos)
}