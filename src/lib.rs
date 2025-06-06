//! Crate-level utilities for working with Databento MBO data.
//!
//! This crate focuses on high-speed reconstruction of limit order books.
//! The main functionality lives in the [`lob`] module which defines
//! [`lob::Book`] and [`lob::Market`] for tracking orders and computing
//! quotes.
//! Additional helper modules may be added over time.

pub mod lob;          // order book logic
pub mod features;    // feature extraction utilities
