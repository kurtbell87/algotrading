# Performance Optimization Plan

## Current Performance: 3.1M events/s (17% of 18M target)
## Target: 15M+ events/s (85%+ of 18M target)

## Critical Bottlenecks Identified

### 1. **MboMsg Cloning (2-3x impact)** - HIGHEST PRIORITY
- **Problem**: `batch.push(rec.clone())` in reader.rs:73
- **Impact**: Every event gets deep copied
- **Fix**: Zero-copy processing with references

### 2. **Lock Contention (1.5-2x impact)** - HIGH PRIORITY  
- **Problem**: Multiple locks per event in strategy context
- **Impact**: Serializes parallel strategy processing
- **Fix**: Lockless data structures or pre-computed snapshots

### 3. **String Allocations in Features (1.5x impact)** - HIGH PRIORITY
- **Problem**: String creation on every feature add
- **Impact**: Heap allocations in hot path
- **Fix**: Pre-allocated feature indices

### 4. **Order Book Linear Search (1.2x impact)** - MEDIUM PRIORITY
- **Problem**: O(n) order lookups on modifications
- **Impact**: Scales poorly with order count
- **Fix**: HashMap for O(1) order lookups

## Implementation Strategy

### Phase 1: Zero-Copy Event Processing
```rust
// Before (cloning):
batch.push(rec.clone()); // 2-3x slowdown

// After (zero-copy):
batch.push(rec); // or process in-place
```

### Phase 2: Lockless Strategy Context
```rust
// Before (locks):
let mut books = self.order_books.write().unwrap();

// After (snapshots):
pub struct MarketSnapshot {
    bbo: HashMap<InstrumentId, (Price, Price)>,
    last_trades: HashMap<InstrumentId, Price>,
}
```

### Phase 3: Indexed Feature System
```rust
// Before (strings):
features.add("spread_absolute", value); // String allocation

// After (indices):
features[SPREAD_ABS_IDX] = value; // Array access
```

### Phase 4: O(1) Order Lookup
```rust
// Before (linear):
level.iter().position(|o| o.order_id == id)

// After (hash):
order_lookup.get(&order_id).map(|(level_idx, order_idx)| ...)
```

## Expected Performance Improvements

| Optimization | Current Impact | After Fix | Cumulative Gain |
|--------------|---------------|-----------|-----------------|
| Remove Cloning | 3.1M events/s | 7-9M events/s | 2.3-2.9x |
| Lockless Context | 7-9M events/s | 10-15M events/s | 1.4-1.7x |
| Indexed Features | 10-15M events/s | 12-16M events/s | 1.2x |
| O(1) Order Lookup | 12-16M events/s | 13-17M events/s | 1.1x |

**Total Expected**: 13-17M events/s (72-94% of 18M target)

## Additional Optimizations

### Memory Pool Allocation
- Pre-allocate event buffers
- Reuse order structures
- Pool strategy contexts

### SIMD Optimizations
- Vectorized price calculations
- Parallel feature computation
- Batch order processing

### CPU Cache Optimization
- Struct layout optimization
- Data locality improvements
- Prefetching hints

### Profile-Guided Optimization
- Identify true hot paths with profiling
- Optimize based on actual usage patterns
- Fine-tune for specific workloads

## Next Steps

1. **Immediate**: Implement zero-copy event processing
2. **Short-term**: Add lockless market snapshots  
3. **Medium-term**: Indexed feature system
4. **Long-term**: Full memory pooling and SIMD

This should get us from 3.1M → 13-17M events/s, achieving 72-94% of the 18M target.