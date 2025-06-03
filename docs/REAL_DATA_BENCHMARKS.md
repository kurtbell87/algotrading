# Real Data Performance Benchmarks

This document describes the real market data benchmarks implemented in `benches/system_performance.rs`.

## Overview

The benchmarks use actual MBO (Market by Order) data files from CME Globex instead of synthetic data, providing realistic performance measurements for the algorithmic trading system.

## Data Source

- **Location**: `/Users/brandonbell/LOCAL_DEV/Market_Data/GLBX-20250528-84NHYCGUFY/`
- **Format**: Databento MBO compressed with zstd (`.mbo.dbn.zst` files)
- **Exchange**: CME Globex (GLBX)
- **Instrument**: MES Futures (ID: 5921)
- **Period**: April-May 2025 (26 trading days)
- **Message Count**: ~12M messages per day file

## Benchmark Components

### 1. Raw MBO File Reading (`mbo_file_reading`)
Tests the performance of reading and decompressing MBO data files:
- Measures throughput in messages/second
- Tests with 1, 5, and 10 files
- Uses memory-mapped I/O and producer-consumer pattern
- Target: >10M messages/second

### 2. Order Book Reconstruction (`order_book_reconstruction`)
Benchmarks order book building from MBO events:
- Single instrument book updates
- Multi-instrument market aggregation
- Pre-loads 1M messages for consistent testing
- Target: >5M updates/second

### 3. Strategy Processing (`strategy_real_data`)
Tests strategy execution on real market events:
- Mean reversion strategy with realistic parameters
- Processes BBO updates and trades from real data
- Generates orders based on actual market conditions
- Target: >1M events/second

### 4. Full Backtesting Engine (`backtest_engine_real_data`)
Complete backtesting simulation:
- Tests with 1, 2, and 5 files to show scalability
- Includes position tracking and risk management
- Realistic fill models and latency simulation
- Limits to 500K events per run for reasonable benchmark time
- Target: >500K events/second

### 5. Feature Extraction (`feature_extraction_real_data`)
Benchmarks feature calculation from order book states:
- Extracts microstructure features
- Tests all feature modules (book, flow, rolling, time, context)
- Target: >100K calculations/second

## Performance Targets

Based on the 18M msg/sec ceiling mentioned in requirements:
- File Reading: >10M messages/second (55% of ceiling)
- Order Book: >5M updates/second (28% of ceiling)
- Strategy: >1M events/second (5.5% of ceiling)
- Backtesting: >500K events/second (2.8% of ceiling)
- Features: >100K calculations/second (0.6% of ceiling)

## Key Optimizations

1. **Memory-mapped I/O**: Direct file access without copying data
2. **Producer-consumer pattern**: 4KB batch processing for efficiency
3. **Zero-copy processing**: Minimal data copying where possible
4. **Fixed-point arithmetic**: Integer-based price calculations
5. **Efficient data structures**: BTreeMap for sorted prices, HashMap for O(1) lookups

## Running the Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench mbo_file_reading

# Quick test run
cargo bench --bench system_performance -- --test

# Generate detailed HTML report
cargo bench --bench system_performance -- --save-baseline real_data
```

## Interpreting Results

The benchmarks report:
- **Throughput**: Messages/events processed per second
- **Time**: Average time per iteration
- **Sample size**: Number of iterations for statistical significance
- **Variance**: Consistency of performance

Compare results against the performance targets to ensure the system meets requirements.

## Notes

- The "disconnected channel" warnings during benchmarking are normal and occur when the file reader thread completes before all messages are consumed
- Benchmark times increase with file count due to I/O and decompression overhead
- Feature extraction benchmarks may show 0 book states if the order book isn't properly initialized - this is a limitation of the current Book API that expects MboMsg directly