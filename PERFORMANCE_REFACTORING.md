# Backtesting Performance Refactoring Plan

## Current Performance Analysis

Based on testing with real MBO data files:
- **File reading**: 18M events/s ✅ (already optimal)
- **Full backtest**: 3.6M events/s ❌ (5x slowdown)
- **Target**: >15M events/s (>83% of 18M)

## Key Bottlenecks Identified

1. **Event Queue Overhead** (30-40% impact)
   - BinaryHeap with PrioritizedEvent
   - Every event gets pushed/popped
   - Sorting overhead for priority

2. **Lock Contention** (20-30% impact)
   - RwLock on market state
   - Multiple lock acquisitions per event
   - Blocks parallel processing

3. **Memory Allocations** (20% impact)
   - Vec allocations for strategy outputs
   - String allocations in features
   - Event cloning

4. **Strategy Dispatch** (10% impact)
   - HashMap lookups
   - Dynamic dispatch overhead

## Optimization Strategy

### Phase 1: Direct Processing (Eliminate Event Queue)
```rust
// Instead of:
while let Some(update) = reader.next_update() {
    let event = convert_to_event(update);
    event_queue.push(PrioritizedEvent { event, ... });
}
while let Some(event) = event_queue.pop() {
    process_event(event);
}

// Use direct processing:
while let Some(update) = reader.next_update() {
    process_update_directly(update);
}
```
**Expected gain**: 30-40% improvement

### Phase 2: Batch Processing
```rust
const BATCH_SIZE: usize = 1000;
let mut batch = Vec::with_capacity(BATCH_SIZE);

// Read batch
while batch.len() < BATCH_SIZE {
    if let Some(update) = reader.next_update() {
        batch.push(update);
    } else { break; }
}

// Process batch with single lock
{
    let mut market_state = self.market_state.write().unwrap();
    for update in &batch {
        market_state.update(update);
    }
}

// Process strategies
for update in &batch {
    for strategy in &mut strategies {
        strategy.process(update);
    }
}
```
**Expected gain**: 20-30% improvement (better cache locality)

### Phase 3: Pre-allocated Buffers
```rust
struct Engine {
    // Pre-allocate all buffers
    event_buffer: Vec<MarketUpdate>,
    order_buffer: Vec<OrderRequest>,
    output_buffer: Vec<StrategyOutput>,
}

// Reuse buffers
self.order_buffer.clear();
for order in &output.orders {
    self.order_buffer.push(order.clone());
}
```
**Expected gain**: 15-20% improvement

### Phase 4: Lockless Market State
```rust
// Instead of RwLock, use atomic operations or 
// single-threaded processing with direct access
struct FastMarketState {
    last_prices: HashMap<InstrumentId, Price>,
    // No locks needed in single-threaded context
}
```
**Expected gain**: 20% improvement

## Implementation Approach

1. **Minimal Changes First**
   - Start with batch processing in existing engine
   - Pre-allocate buffers
   - Measure impact

2. **Gradual Refactoring**
   - Create engine_batch.rs with batch processing
   - Keep API compatibility
   - A/B test performance

3. **Advanced Optimizations** (if needed)
   - SIMD for price calculations
   - Custom allocators
   - Zero-copy event processing

## Expected Total Performance

Combining optimizations:
- Base: 3.6M events/s
- With optimizations: 10-15M events/s (2.8-4.2x improvement)
- Should achieve 55-83% of 18M target

## Quick Win Implementation

The fastest improvement would be batch processing in the existing engine:

```rust
// In engine.rs process_market_event()
fn process_market_events_batch(&mut self, events: &[MarketEvent]) {
    // Single lock for batch
    {
        let mut state = self.market_state.write().unwrap();
        for event in events {
            state.process_event(event);
        }
    }
    
    // Process all strategies
    for event in events {
        // Existing strategy processing
    }
}
```

This simple change could yield 30-50% improvement with minimal risk.