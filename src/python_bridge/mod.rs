//! Python integration bridge for ML models and strategies
//!
//! This module provides PyO3-based integration allowing:
//! - Embedding Python ML models in Rust strategies
//! - Writing strategies in Python that integrate with the Rust engine
//! - Efficient data marshaling between Rust and Python

pub mod types;
pub mod models;
pub mod strategy;

pub use types::*;
pub use models::*;
pub use strategy::*;