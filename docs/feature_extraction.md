# Feature Extraction Plan

This document outlines an approach for converting raw market data from the
Databento DBN files into features that can be consumed by downstream
machine learning or reinforcement learning models.

## 1. Goals

- Provide a repeatable pipeline for turning MBO messages into numerical
  representations.
- Preserve the high throughput design of the existing system.
- Output features in a format easily loaded by Python-based ML/RL tools
  (e.g. `numpy` arrays or Parquet files).

## 2. Raw Data Overview

The repository currently processes Databento Market By Order messages as
seen in the [`Market`](../src/lob.rs) implementation. Each `MboMsg` is
applied to a [`Book`] which tracks individual orders and aggregates best
bid and ask levels.

Example fields available per message include:

- `price` and `size`
- `side` (`Bid` or `Ask`)
- `action` (`Add`, `Modify`, `Cancel`, `Clear`)
- timestamps such as `ts_recv` and `ts_in_delta`

The test data in `test_data/*.dbn.zst` provides realistic CME Globex
order flow for experimentation.

## 3. Proposed Features

Below are candidate features for per-tick or windowed extraction.
These can be adjusted based on modelling requirements:

1. **Mid Price and Spread**
   - Mid = `(best_bid + best_ask) / 2`
   - Spread = `best_ask - best_bid`
2. **Depth by Level**
   - Sizes at the top `N` bid and ask prices (e.g. top 5 levels).
   - Order counts per level (from `LevelSummary::count`).
3. **Order Flow Metrics**
   - Net order additions vs cancellations within a time window.
   - Volume at price changes and trade aggressor direction (if trade messages are added).
4. **Imbalance Indicators**
   - `(bid_size - ask_size) / (bid_size + ask_size)` for top levels.
5. **Time Features**
   - Normalized time of day or elapsed time since session start.
6. **Publisher Distribution**
   - Stats per publisher when multiple books exist (see `Market::aggregated_bbo`).

These features could be sampled every message, every `N` messages, or on a fixed
time grid depending on training needs.

## 4. Output Format

To keep integration with Python simple, a binary format such as Parquet or a
columnar NumPy `.npz` file is recommended. Each record would contain the
features above plus the target variable (if applicable).

Example structure:

```text
timestamp, mid_px, spread, bid1_sz, ask1_sz, ...
```

## 5. Implementation Sketch

1. Extend the consumer loop in `src/main.rs` to periodically snapshot the
   [`Market`] state.
2. Convert the snapshot into feature vectors using the calculations in
   this document.
3. Batch-write features to disk to minimize I/O overhead.
4. Provide a small Python script in `examples/` to load the output and
   demonstrate basic usage.

## 6. Next Steps

- Decide on sampling frequency and windowing strategy.
- Prototype the feature extraction on the included test data.
- Benchmark throughput to ensure we maintain performance goals outlined in
  the README (e.g. "~55 nanoseconds per message processing time"【F:README.md†L68-L71】).

