# High-Performance Algorithmic Trading System

A Rust-based algorithmic trading system optimized for processing market microstructure data at hardware limits. Achieves 3.55M events/second for full strategy backtesting and 17.86M events/second for order book reconstruction.

## Features

- **Ultra-fast MBO processing**: Handles CME Globex Market-by-Order data at near-hardware speeds
- **Full order book reconstruction**: BTreeMap/HashMap hybrid for optimal performance
- **Backtesting engine**: Event-driven simulation with accurate position tracking
- **Strategy framework**: Extensible architecture for custom trading strategies
- **Python integration**: Bridge for ML model inference via PyO3
- **Feature extraction**: Pre-compute market microstructure features in parallel

## Performance

- **Order Book Reconstruction**: 17.86M events/sec (99% of theoretical max)
- **Full Backtesting**: 3.55M events/sec (843 CPU cycles/event)
- **5-Year Backtest**: ~3.5 hours for 15B events

The system operates at hardware limits, using less than 1000 CPU cycles per event.

## Quick Start

```bash
# Build the project
cargo build --release

# Run on market data directory
cargo run --release ../Market_Data/GLBX-20250528-84NHYCGUFY/

# Run in verify mode (downloads sample data)
cargo run --release -- --verify

# Run benchmarks
cargo bench
```

## Architecture

The system uses a modular architecture with clear separation of concerns:

```
algotrading/
├── core/           # Core types and traits
├── market_data/    # MBO data ingestion
├── order_book/     # Order book reconstruction
├── features/       # Feature extraction
├── strategies/     # Trading strategies
├── backtest/       # Backtesting engine
└── python_bridge/  # Python/ML integration
```

Key design decisions:
- Memory-mapped I/O for zero-copy file reading
- Producer-consumer pattern with 4KB batching
- Lock-free channels for thread communication
- Fixed-point arithmetic (i64 with 9 decimal places)

## Trading Strategies

Built-in strategies:
- **Mean Reversion**: Trade on deviations from rolling mean
- **Market Making**: Provide liquidity with dynamic quotes
- **Trend Following**: Momentum-based directional trading

Custom strategies implement the `Strategy` trait:

```rust
pub trait Strategy: Send {
    fn initialize(&mut self, context: &StrategyContext) -> Result<(), String>;
    fn on_market_event(&mut self, event: &MarketEvent, context: &StrategyContext) -> StrategyOutput;
    fn on_timer(&mut self, timestamp: u64, context: &StrategyContext) -> StrategyOutput;
    fn on_fill(&mut self, fill: &FillEvent, context: &StrategyContext);
}
```

## Python Integration

Train ML models in Python and use them in Rust strategies:

```bash
cd python_ml/
uv sync                    # Install dependencies
python train_model.py      # Train example model
python python_strategy_example.py  # Test the strategy
```

## Benchmarking

Comprehensive benchmarks are included:

```bash
# Run all benchmarks
cargo bench

# Profile backtest performance
cargo run --release --bin profile_all_events

# Compare strategies
cargo run --release --bin strategy_backtest_benchmark
```

## Data Format

The system processes Databento MBO (Market by Order) files in `.dbn.zst` format. Each file contains a full day of tick-by-tick market data with nanosecond timestamps.

## Documentation

- [Architecture](ARCHITECTURE.md) - System design and components
- [Performance Analysis](PERFORMANCE_FINAL.md) - Detailed performance metrics
- [Trading Strategies](docs/TRADING_STRATEGIES.md) - Strategy implementation guide
- [Backtesting](docs/BACKTESTING_ARCHITECTURE.md) - Backtesting engine design

## Requirements

- Rust 1.70+
- Python 3.8+ (for ML integration)
- 8GB+ RAM recommended
- Unix-like OS (Linux/macOS)

## License

This project is proprietary software. All rights reserved.