//! src/main.rs
//! • default: process market data files from a directory or single file
//! • --verify: download GOOG/GOOGL sample and verify processing

mod verify;

use algotrading::{
    core::{MarketDataSource, MarketUpdate},
    market_data::FileReader,
    order_book::Book,
};
use std::{
    collections::HashMap,
    env,
    ffi::OsStr,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::{Duration as StdDuration, Instant},
};

type InstId = u32;

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
            eprintln!("USAGE\n  cargo run --release <file|dir>\n  cargo run --release -- --verify");
            std::process::exit(1);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Main processing                                                    */
/* ------------------------------------------------------------------ */

fn process_market_data(paths: Vec<PathBuf>) -> std::result::Result<(), Box<dyn std::error::Error>> {
    process_market_data_with_callback(paths, |_, _, _| {})
}

fn process_market_data_with_callback<F>(
    paths: Vec<PathBuf>,
    mut callback: F,
) -> std::result::Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(&MarketUpdate, &mut HashMap<InstId, Book>, u64),
{
    let start = Instant::now();
    let mut reader = FileReader::new(paths)?;
    let mut books = HashMap::<InstId, Book>::new();
    let mut n = 0u64;
    let mut next_print = Instant::now();
    let mut spinner_anim_idx = 0;

    static SPINNER: &[char] = &['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'];

    while let Some(update) = reader.next_update() {
        n += 1;

        match &update {
            MarketUpdate::OrderBook(book_update) => {
                // For now, just count - we'd apply to books here
                books.entry(book_update.instrument_id).or_default();
            }
            MarketUpdate::Trade(_trade) => {
                // Handle trades
            }
        }

        // Call the callback with the update
        callback(&update, &mut books, n);

        if next_print.elapsed() >= StdDuration::from_millis(120) {
            next_print = Instant::now();
            let frame = SPINNER[spinner_anim_idx % SPINNER.len()];
            spinner_anim_idx += 1;

            print!(
                "\r\x1B[33m{frame} {n:>13} msgs\x1B[0m \
                \x1B[32m| {:>4} instruments\x1B[0m",
                books.len()
            );
            io::stdout().flush().ok();
        }
    }

    println!(); // finish the spinner line

    let elapsed = start.elapsed();
    let nanos_per_msg = elapsed.as_nanos() / n as u128;
    let msgs_per_sec = 1_000_000_000.0 / nanos_per_msg as f64;

    println!(
        "decoded {} msgs in {:.3}s → {:.1} ns/msg → {:.1} msgs/s",
        n,
        elapsed.as_secs_f64(),
        nanos_per_msg,
        msgs_per_sec
    );

    Ok(())
}

/* ------------------------------------------------------------------ */
/*  Verify mode                                                        */
/* ------------------------------------------------------------------ */

async fn verify_mode() -> std::result::Result<(), Box<dyn std::error::Error>> {
    verify::run_verify_mode().await
}

/* ------------------------------------------------------------------ */
/*  Main entry point                                                   */
/* ------------------------------------------------------------------ */

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    match parse_args() {
        Mode::Speed(paths) => process_market_data(paths),
        Mode::Verify => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(verify_mode())
        }
    }
}
