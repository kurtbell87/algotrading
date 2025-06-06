# Test Market Data

This directory contains real market data files from Databento for testing and development purposes.

## Files

### Market Data (Git LFS)
- `glbx-mdp3-20250501.mbo.dbn.zst` - May 1, 2025 (229 MB)
- `glbx-mdp3-20250502.mbo.dbn.zst` - May 2, 2025 (218 MB)
- `glbx-mdp3-20250505.mbo.dbn.zst` - May 5, 2025 (159 MB)

### Metadata Files
- `symbology.json` - Symbol mappings and instrument definitions
- `metadata.json` - Dataset metadata including schema, time ranges, and data sources
- `condition.json` - Trading condition codes and their meanings

## Data Format

These files contain Market By Order (MBO) data from CME Globex:
- Schema: MBO (Market By Order) - order-level granularity
- Exchange: GLBX (CME Globex)
- Protocol: MDP3
- Compression: zstd
- Time Period: Three sequential trading days in May 2025

## Usage

The large `.dbn.zst` files are tracked using Git LFS. To work with them:

1. Ensure Git LFS is installed: `git lfs install`
2. Clone or pull to automatically download the files
3. Use the databento crate to decode:

```rust
use databento::dbn::{decode::Decoder, RecordRef};

// Decode a file
let mut decoder = Decoder::from_zstd_file("test_data/glbx-mdp3-20250501.mbo.dbn.zst")?;

// Read the symbology mapping
let symbology = std::fs::read_to_string("test_data/symbology.json")?;

// Process records
while let Some(rec) = decoder.decode_record()? {
    match rec {
        RecordRef::Mbo(mbo) => {
            // Process MBO record
        }
        _ => {}
    }
}
```

## Purpose

This test data enables:
- Realistic testing of order book reconstruction
- Performance benchmarking with production-scale data
- Integration testing with actual market data formats
- Development without requiring API access

## Source

Data provided by Databento (https://databento.com) for CME Globex futures market.