use crate::core::{MarketDataSource, MarketUpdate, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use databento::dbn::{
    decode::{DbnDecoder, DecodeRecord},
    record::MboMsg,
};
use memmap2::Mmap;
use std::{
    fs::File,
    io::BufReader,
    path::PathBuf,
    thread,
};

const BATCH_SIZE: usize = 4 * 1024;

/// File-based market data reader
pub struct FileReader {
    rx: Receiver<Option<Vec<MboMsg>>>,
    _handle: thread::JoinHandle<()>,
    current_batch: Vec<MboMsg>,
    batch_index: usize,
}

impl FileReader {
    /// Create a new file reader for the given paths
    pub fn new(paths: Vec<PathBuf>) -> Result<Self> {
        let (tx, rx) = bounded(10); // Buffer up to 10 batches
        
        let handle = thread::spawn(move || {
            Self::producer_thread(paths, tx);
        });

        Ok(Self {
            rx,
            _handle: handle,
            current_batch: Vec::new(),
            batch_index: 0,
        })
    }

    /// Producer thread that reads files and sends batches
    fn producer_thread(paths: Vec<PathBuf>, tx: Sender<Option<Vec<MboMsg>>>) {
        let mut batch = Vec::with_capacity(BATCH_SIZE);

        for path in paths {
            if let Err(e) = Self::process_file(&path, &tx, &mut batch) {
                eprintln!("Error processing file {:?}: {}", path, e);
            }
        }

        // Send any remaining messages
        if !batch.is_empty() {
            tx.send(Some(batch)).ok();
        }
        
        // Signal end of stream
        tx.send(None).ok();
    }

    /// Process a single file
    fn process_file(
        path: &PathBuf,
        tx: &Sender<Option<Vec<MboMsg>>>,
        batch: &mut Vec<MboMsg>,
    ) -> Result<()> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let reader = BufReader::new(std::io::Cursor::new(&mmap[..]));
        let mut decoder = DbnDecoder::with_zstd_buffer(reader)?;

        while let Some(rec) = decoder.decode_record::<MboMsg>()? {
            batch.push(rec.clone());
            if batch.len() == BATCH_SIZE {
                tx.send(Some(std::mem::replace(batch, Vec::with_capacity(BATCH_SIZE))))?;
            }
        }

        Ok(())
    }
}

impl MarketDataSource for FileReader {
    fn subscribe(&mut self, _instruments: Vec<crate::core::InstrumentId>) -> Result<()> {
        // File reader doesn't support filtering by instrument
        Ok(())
    }

    fn next_update(&mut self) -> Option<MarketUpdate> {
        // Check if we need to fetch a new batch
        if self.batch_index >= self.current_batch.len() {
            match self.rx.recv().ok()? {
                Some(batch) => {
                    self.current_batch = batch;
                    self.batch_index = 0;
                }
                None => return None, // End of stream
            }
        }

        // Get next message from current batch
        let msg = &self.current_batch[self.batch_index];
        self.batch_index += 1;

        // Convert to MarketUpdate
        Some(crate::market_data::normalizer::mbo_to_market_update(msg))
    }
}