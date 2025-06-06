# Algotrading

A high-performance order book reconstruction and market data processing system written in Rust.

## Features

- Ultra-fast market data processing (>17M messages/second)
- Multi-threaded architecture with producer-consumer pattern
- Memory-mapped file I/O for optimal performance
- Support for Databento DBN format with zstd compression
- Real-time order book reconstruction with Market By Order (MBO) data

## Prerequisites

- Rust 1.70 or later
- Git LFS (Large File Storage) for market data files
- Databento API key (for downloading additional data)

## Setup

1. Install Git LFS if not already installed:
   ```bash
   git lfs install
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd algotrading
   ```

3. The market data files will be downloaded automatically via Git LFS.

4. Set up your Databento API key (optional, for downloading new data):
   ```bash
   echo "DATABENTO_API_KEY=your_key_here" > ../.env
   ```

## Usage

### Running with test data

Process a single file:
```bash
cargo run --release test_data/glbx-mdp3-20250501.mbo.dbn.zst
```

Process all files in a directory:
```bash
cargo run --release test_data
```

### Verify mode

Download and process a sample GOOG/GOOGL dataset:
```bash
cargo run --release -- --verify
```

## Data Files

This repository uses Git LFS to manage large market data files:
- `dbeq-basic-20240403.mbo.dbn.zst` - Sample GOOG/GOOGL data for verification
- `test_data/*.dbn.zst` - CME Globex futures market data for testing

## Performance

The system achieves:
- ~55 nanoseconds per message processing time
- >17 million messages per second throughput
- Efficient memory usage through mmap and batch processing

## Architecture

- **Producer thread**: Memory-maps files and decodes compressed market data
- **Consumer thread**: Processes messages and maintains order books
- **Channel-based communication**: Lock-free message passing between threads
- **Batch processing**: Messages processed in batches of 4096 for cache efficiency

## License

[Your license here]