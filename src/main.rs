//! src/main.rs
//!   • default  : mmap + two-thread replay of a .dbn.zst **file** *or* all files
//!                in a directory
//!   • --verify : download GOOG/GOOGL sample and print the first 7 documented lines

use std::{
    collections::HashMap,
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Instant, Duration},
    io::{self, Write},
};

use serde_json::Value;

use crossbeam_channel::{bounded, Receiver};
use databento::{
    dbn::{
        decode::{AsyncDbnDecoder, DbnDecoder, DecodeRecord},
        record::MboMsg,
        Dataset, Schema, SymbolIndex
    },
    historical::timeseries::GetRangeToFileParams,
    HistoricalClient
};
use memmap2::Mmap;
use time::macros::datetime;

mod lob;
use lob::{Book, Market};

type InstId = u32;
const BATCH: usize = 4 * 1024;

/* ------------------------------------------------------------------ */
/*  Mode flags                                                         */
/* ------------------------------------------------------------------ */

enum Mode {
    Speed(Vec<PathBuf>), // one or many .dbn.zst files
    Verify,
}

fn parse_args() -> Mode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("--verify") => Mode::Verify,
        Some(p) => {
            let path = Path::new(p);
            if path.is_dir() {
                // gather *.dbn.zst inside the directory (non-recursive)
                let mut files: Vec<PathBuf> = fs::read_dir(path)
                    .expect("dir")
                    .filter_map(|e| {
                        let p = e.ok()?.path();
                        (p.extension() == Some(OsStr::new("zst"))).then_some(p)
                    })
                    .collect();
                files.sort(); // lexicographic (== chronological for your filenames)
                if files.is_empty() {
                    eprintln!("Directory has no *.dbn.zst files");
                    std::process::exit(1);
                }
                Mode::Speed(files)
            } else {
                Mode::Speed(vec![path.to_path_buf()])
            }
        }
        _ => {
            eprintln!(
                "USAGE\n  cargo run --release <file|dir>\n  cargo run --release -- --verify"
            );
            std::process::exit(1);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Producer thread                                                    */
/* ------------------------------------------------------------------ */

fn spawn_producer(paths: Vec<PathBuf>, tx: crossbeam_channel::Sender<Option<Vec<MboMsg>>>) {
    thread::spawn(move || {
        let mut batch = Vec::with_capacity(BATCH);

        for path in paths {
            let f = std::fs::File::open(&path).expect("open");
            let mmap = unsafe { Mmap::map(&f).expect("mmap") };
            let reader = std::io::BufReader::new(std::io::Cursor::new(&mmap[..]));
            let mut dec = DbnDecoder::with_zstd_buffer(reader).expect("decoder");

            while let Some(rec) = dec.decode_record::<MboMsg>().expect("decode") {
                batch.push(rec.clone());
                if batch.len() == BATCH {
                    tx.send(Some(batch)).unwrap();
                    batch = Vec::with_capacity(BATCH);
                }
            }
        }

        if !batch.is_empty() {
            tx.send(Some(batch)).ok();
        }
        tx.send(None).ok();
    });
}

/* ------------------------------------------------------------------ */
/*  Consumer                                                           */
/* ------------------------------------------------------------------ */

fn consume(rx: Receiver<Option<Vec<MboMsg>>>) -> (u64, HashMap<InstId, Book>) {
    let mut books = HashMap::<InstId, Book>::new();
    let mut n = 0u64;
    let mut next_print = Instant::now();
    let mut spinner_anim_idx = 0;
    
    while let Ok(opt) = rx.recv() {
        let chunk = match opt { Some(c) => c, None => break };
        n += chunk.len() as u64;
        for msg in chunk {
            books.entry(msg.hd.instrument_id).or_default().apply(msg);
        }

        static SPINNER: &[char] = &[
            '⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷',
        ];
        if next_print.elapsed() >= Duration::from_millis(120) { // Or Duration::from_millis(288) for ~2.3s/rev
            next_print = Instant::now();
            
            let frame = SPINNER[spinner_anim_idx % SPINNER.len()];
            spinner_anim_idx += 1; // Always advance animation frame
    
            print!(
                "\r\x1B[33m{frame} {n:>13} msgs\x1B[0m \
                \x1B[32m| {:>4} instruments\x1B[0m",
                books.len()
            );
            io::stdout().flush().ok();
        }
    }
    println!();          // finish the spinner line
    (n, books)
}


/* ------------------------------------------------------------------ */
/*  Verify helper (unchanged)                                          */
/* ------------------------------------------------------------------ */

async fn ensure_sample(path: &str) -> databento::Result<()> {
    if Path::new(path).exists() {
        return Ok(());
    }
    let mut client = HistoricalClient::builder().key_from_env()?.build()?;
    client
        .timeseries()
        .get_range_to_file(
            &GetRangeToFileParams::builder()
                .dataset(Dataset::DbeqBasic)
                .symbols(vec!["GOOG", "GOOGL"])
                .date_time_range((
                    datetime!(2024-04-03 08:00:00 UTC),
                    datetime!(2024-04-03 14:00:00 UTC),
                ))
                .schema(Schema::Mbo)
                .path(path)
                .build(),
        )
        .await?;
    Ok(())
}

/* ------------------------------------------------------------------ */

/// Load symbology mapping from instrument_id to symbol name
fn load_symbology(dir: &Path) -> HashMap<u32, String> {
    let mut symbol_map = HashMap::new();
    let symbology_path = dir.join("symbology.json");
    
    if let Ok(content) = fs::read_to_string(&symbology_path) {
        if let Ok(json) = serde_json::from_str::<Value>(&content) {
            if let Some(result) = json.get("result").and_then(|r| r.as_object()) {
                for (symbol, entries) in result {
                    if let Some(arr) = entries.as_array() {
                        for entry in arr {
                            if let Some(inst_id_str) = entry.get("s").and_then(|s| s.as_str()) {
                                if let Ok(inst_id) = inst_id_str.parse::<u32>() {
                                    symbol_map.insert(inst_id, symbol.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    symbol_map
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env file from parent directory
    if let Err(e) = dotenvy::from_path("../.env") {
        eprintln!("Warning: Could not load ../.env file: {}", e);
    }
    
    match parse_args() {
        /* ----------   VERIFY PATH  --------------------------------- */
        Mode::Verify => {
            let file = "dbeq-basic-20240403.mbo.dbn.zst";
            ensure_sample(file).await?;

            let mut dec = AsyncDbnDecoder::from_zstd_file(file).await?;
            let symbol_map = dec.metadata().symbol_map()?;

            let mut market = Market::default();
            let mut seen = 0usize;

            while let Some(mbo) = dec.decode_record::<MboMsg>().await? {
                market.apply(mbo.clone());
                if mbo.flags.is_last() {
                    seen += 1;
                    let symbol = symbol_map.get_for_rec(mbo).unwrap();
                    let (bb, ba) = market.aggregated_bbo(mbo.hd.instrument_id);
                    println!("{symbol} Aggregated BBO | {}", mbo.ts_recv().unwrap());
                    println!("    {}", ba.map_or("None".into(), |pl| pl.to_string()));
                    println!("    {}", bb.map_or("None".into(), |pl| pl.to_string()));
                }
                if seen >= 7 {
                    break;
                }
            }
        }

        /* ----------  SPEED PATH  ----------------------------------- */
        Mode::Speed(paths) => {
            // Load symbology from the directory containing the files
            let symbol_map = if let Some(parent) = paths.first().and_then(|p| p.parent()) {
                load_symbology(parent)
            } else {
                HashMap::new()
            };
            
            let (tx, rx) = bounded::<Option<Vec<MboMsg>>>(8);
            spawn_producer(paths, tx);

            let t0 = Instant::now();
            let (msgs, books) = consume(rx);
            let dt = t0.elapsed();

            println!("\nFinal Book State:");
            println!("{:-<60}", "");
            
            for (inst_id, book) in &books {
                let (bid, ask) = book.bbo();
                if bid.is_none() && ask.is_none() {
                    continue;
                }
                
                // Get symbol name from mapping or use instrument ID
                let symbol = symbol_map.get(inst_id)
                    .cloned()
                    .unwrap_or_else(|| inst_id.to_string());
                
                println!("{} BBO:", symbol);
                
                // Format ask (top line)
                if let Some(a) = ask {
                    println!("    {:>4} @ {:>15.9} | {:>2} order(s)", 
                        a.size, a.price as f64 / 1_000_000_000.0, a.count);
                } else {
                    println!("    None");
                }
                
                // Format bid (bottom line)
                if let Some(b) = bid {
                    println!("    {:>4} @ {:>15.9} | {:>2} order(s)", 
                        b.size, b.price as f64 / 1_000_000_000.0, b.count);
                } else {
                    println!("    None");
                }
                
                println!();
            }

            println!(
                "decoded {msgs} msgs in {:.3?} → {:.1} ns/msg → {:.1} msgs/s",
                dt,
                dt.as_nanos() as f64 / msgs as f64,
                msgs as f64 / dt.as_secs_f64()
            );
        }
    }
    Ok(())
}
