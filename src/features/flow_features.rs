//! Order flow features
//!
//! This module extracts features from order flow including:
//! - Trade flow imbalance
//! - Aggressive vs passive order ratios
//! - Order arrival rates
//! - Trade size distributions

use crate::core::types::{Price, Quantity};
use crate::features::collector::FeatureVector;
use std::collections::VecDeque;

/// Time window for flow calculations (microseconds)
const DEFAULT_WINDOW_US: u64 = 60_000_000; // 1 minute

/// Trade information for flow calculations
#[derive(Debug, Clone)]
struct TradeInfo {
    timestamp: u64,
    #[allow(dead_code)]
    price: Price,
    quantity: Quantity,
    is_buy: bool,
    is_aggressive: bool,
}

/// Trade flow metrics
#[derive(Debug, Clone, Default)]
pub struct TradeFlowMetrics {
    /// Buy volume in window
    pub buy_volume: f64,
    /// Sell volume in window
    pub sell_volume: f64,
    /// Trade flow imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol)
    pub flow_imbalance: f64,
    /// Net order flow: buy_vol - sell_vol
    pub net_flow: f64,
    /// Trade count in window
    pub trade_count: u32,
    /// Average trade size
    pub avg_trade_size: f64,
}

/// Aggressive/Passive order metrics
#[derive(Debug, Clone, Default)]
pub struct AggressivePassiveMetrics {
    /// Aggressive buy volume
    pub aggressive_buy_volume: f64,
    /// Passive buy volume
    pub passive_buy_volume: f64,
    /// Aggressive sell volume
    pub aggressive_sell_volume: f64,
    /// Passive sell volume
    pub passive_sell_volume: f64,
    /// Aggressive ratio: aggressive / (aggressive + passive)
    pub aggressive_ratio: f64,
    /// Buy aggression ratio: aggressive_buy / total_buy
    pub buy_aggression_ratio: f64,
    /// Sell aggression ratio: aggressive_sell / total_sell
    pub sell_aggression_ratio: f64,
}

/// Order arrival rate metrics
#[derive(Debug, Clone, Default)]
pub struct ArrivalRateMetrics {
    /// Buy order arrival rate (orders per second)
    pub buy_arrival_rate: f64,
    /// Sell order arrival rate (orders per second)
    pub sell_arrival_rate: f64,
    /// Total arrival rate
    pub total_arrival_rate: f64,
    /// Inter-arrival time statistics (microseconds)
    pub avg_inter_arrival_time: f64,
    /// Arrival rate volatility
    pub arrival_rate_volatility: f64,
}

/// Trade size distribution metrics
#[derive(Debug, Clone, Default)]
pub struct TradeSizeMetrics {
    /// Small trade volume (< 25th percentile)
    pub small_trade_volume: f64,
    /// Medium trade volume (25th-75th percentile)
    pub medium_trade_volume: f64,
    /// Large trade volume (> 75th percentile)
    pub large_trade_volume: f64,
    /// Maximum trade size in window
    pub max_trade_size: f64,
    /// Trade size standard deviation
    pub trade_size_std: f64,
}

/// Order flow features extractor
pub struct FlowFeatures {
    /// Time window for calculations
    window_us: u64,
    /// Recent trades within window
    trades: VecDeque<TradeInfo>,
    /// Trade flow metrics
    flow_metrics: TradeFlowMetrics,
    /// Aggressive/passive metrics
    aggression_metrics: AggressivePassiveMetrics,
    /// Arrival rate metrics
    arrival_metrics: ArrivalRateMetrics,
    /// Trade size metrics
    size_metrics: TradeSizeMetrics,
    /// Last trade timestamp for inter-arrival calculation
    last_trade_time: Option<u64>,
    /// Inter-arrival times for volatility calculation
    inter_arrival_times: VecDeque<f64>,
}

impl FlowFeatures {
    pub fn new() -> Self {
        Self::with_window(DEFAULT_WINDOW_US)
    }

    pub fn with_window(window_us: u64) -> Self {
        Self {
            window_us,
            trades: VecDeque::new(),
            flow_metrics: TradeFlowMetrics::default(),
            aggression_metrics: AggressivePassiveMetrics::default(),
            arrival_metrics: ArrivalRateMetrics::default(),
            size_metrics: TradeSizeMetrics::default(),
            last_trade_time: None,
            inter_arrival_times: VecDeque::with_capacity(100),
        }
    }

    /// Update with new trade
    pub fn update_trade(&mut self, price: Price, quantity: Quantity, is_buy: bool, timestamp: u64) {
        // Determine if trade is aggressive (taking liquidity)
        // In practice, this would be determined by comparing to order book state
        // For now, we'll use a simple heuristic
        let is_aggressive = true; // Market orders are aggressive

        let trade = TradeInfo {
            timestamp,
            price,
            quantity,
            is_buy,
            is_aggressive,
        };

        // Update inter-arrival time
        if let Some(last_time) = self.last_trade_time {
            let inter_arrival = (timestamp - last_time) as f64;
            self.inter_arrival_times.push_back(inter_arrival);
            if self.inter_arrival_times.len() > 100 {
                self.inter_arrival_times.pop_front();
            }
        }
        self.last_trade_time = Some(timestamp);

        // Add trade and remove old trades outside window
        self.trades.push_back(trade);
        self.remove_old_trades(timestamp);

        // Recalculate all metrics
        self.calculate_metrics();
    }

    /// Remove trades outside the time window
    fn remove_old_trades(&mut self, current_time: u64) {
        let cutoff_time = current_time as i64 - self.window_us as i64;
        while let Some(front) = self.trades.front() {
            if (front.timestamp as i64) < cutoff_time {
                self.trades.pop_front();
            } else {
                break;
            }
        }
    }

    /// Calculate all flow metrics
    fn calculate_metrics(&mut self) {
        // Reset metrics
        self.flow_metrics = TradeFlowMetrics::default();
        self.aggression_metrics = AggressivePassiveMetrics::default();
        self.size_metrics = TradeSizeMetrics::default();

        if self.trades.is_empty() {
            self.arrival_metrics = ArrivalRateMetrics::default();
            return;
        }

        // Collect trade sizes for distribution analysis
        let mut trade_sizes: Vec<f64> = Vec::new();

        // Calculate volumes and counts
        for trade in &self.trades {
            let volume = trade.quantity.as_f64();
            trade_sizes.push(volume);

            if trade.is_buy {
                self.flow_metrics.buy_volume += volume;
                if trade.is_aggressive {
                    self.aggression_metrics.aggressive_buy_volume += volume;
                } else {
                    self.aggression_metrics.passive_buy_volume += volume;
                }
            } else {
                self.flow_metrics.sell_volume += volume;
                if trade.is_aggressive {
                    self.aggression_metrics.aggressive_sell_volume += volume;
                } else {
                    self.aggression_metrics.passive_sell_volume += volume;
                }
            }
        }

        // Calculate flow metrics
        self.flow_metrics.trade_count = self.trades.len() as u32;
        let total_volume = self.flow_metrics.buy_volume + self.flow_metrics.sell_volume;

        if total_volume > 0.0 {
            self.flow_metrics.flow_imbalance =
                (self.flow_metrics.buy_volume - self.flow_metrics.sell_volume) / total_volume;
            self.flow_metrics.avg_trade_size = total_volume / self.flow_metrics.trade_count as f64;
        }
        self.flow_metrics.net_flow = self.flow_metrics.buy_volume - self.flow_metrics.sell_volume;

        // Calculate aggression metrics
        let total_aggressive = self.aggression_metrics.aggressive_buy_volume
            + self.aggression_metrics.aggressive_sell_volume;
        let total_passive = self.aggression_metrics.passive_buy_volume
            + self.aggression_metrics.passive_sell_volume;

        if total_aggressive + total_passive > 0.0 {
            self.aggression_metrics.aggressive_ratio =
                total_aggressive / (total_aggressive + total_passive);
        }

        if self.flow_metrics.buy_volume > 0.0 {
            self.aggression_metrics.buy_aggression_ratio =
                self.aggression_metrics.aggressive_buy_volume / self.flow_metrics.buy_volume;
        }

        if self.flow_metrics.sell_volume > 0.0 {
            self.aggression_metrics.sell_aggression_ratio =
                self.aggression_metrics.aggressive_sell_volume / self.flow_metrics.sell_volume;
        }

        // Calculate arrival rates
        self.calculate_arrival_rates();

        // Calculate trade size distribution
        self.calculate_size_distribution(&trade_sizes);
    }

    /// Calculate order arrival rates
    fn calculate_arrival_rates(&mut self) {
        if self.trades.len() < 2 {
            return;
        }

        let time_span = (self.trades.back().unwrap().timestamp
            - self.trades.front().unwrap().timestamp) as f64
            / 1_000_000.0; // Convert to seconds

        if time_span > 0.0 {
            let buy_count = self.trades.iter().filter(|t| t.is_buy).count() as f64;
            let sell_count = self.trades.iter().filter(|t| !t.is_buy).count() as f64;

            self.arrival_metrics.buy_arrival_rate = buy_count / time_span;
            self.arrival_metrics.sell_arrival_rate = sell_count / time_span;
            self.arrival_metrics.total_arrival_rate = self.trades.len() as f64 / time_span;
        }

        // Calculate inter-arrival statistics
        if !self.inter_arrival_times.is_empty() {
            let sum: f64 = self.inter_arrival_times.iter().sum();
            let count = self.inter_arrival_times.len() as f64;
            self.arrival_metrics.avg_inter_arrival_time = sum / count;

            // Calculate volatility (standard deviation)
            let mean = self.arrival_metrics.avg_inter_arrival_time;
            let variance = self
                .inter_arrival_times
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / count;
            self.arrival_metrics.arrival_rate_volatility = variance.sqrt();
        }
    }

    /// Calculate trade size distribution metrics
    fn calculate_size_distribution(&mut self, trade_sizes: &[f64]) {
        if trade_sizes.is_empty() {
            return;
        }

        // Sort sizes for percentile calculation
        let mut sorted_sizes = trade_sizes.to_vec();
        sorted_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted_sizes.len();
        let p25_idx = len / 4;
        let p75_idx = 3 * len / 4;

        let p25 = sorted_sizes[p25_idx];
        let p75 = sorted_sizes[p75_idx];

        // Categorize trades by size
        for (i, _trade) in self.trades.iter().enumerate() {
            let size = trade_sizes[i];
            if size < p25 {
                self.size_metrics.small_trade_volume += size;
            } else if size > p75 {
                self.size_metrics.large_trade_volume += size;
            } else {
                self.size_metrics.medium_trade_volume += size;
            }
        }

        self.size_metrics.max_trade_size = sorted_sizes[len - 1];

        // Calculate standard deviation
        let mean = trade_sizes.iter().sum::<f64>() / len as f64;
        let variance = trade_sizes.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / len as f64;
        self.size_metrics.trade_size_std = variance.sqrt();
    }

    /// Add features to feature vector
    pub fn add_to_vector(&self, features: &mut FeatureVector) {
        // Flow metrics
        features.add("flow_buy_volume", self.flow_metrics.buy_volume);
        features.add("flow_sell_volume", self.flow_metrics.sell_volume);
        features.add("flow_imbalance", self.flow_metrics.flow_imbalance);
        features.add("flow_net", self.flow_metrics.net_flow);
        features.add("flow_trade_count", self.flow_metrics.trade_count as f64);
        features.add("flow_avg_trade_size", self.flow_metrics.avg_trade_size);

        // Aggression metrics
        features.add("aggression_ratio", self.aggression_metrics.aggressive_ratio);
        features.add(
            "buy_aggression_ratio",
            self.aggression_metrics.buy_aggression_ratio,
        );
        features.add(
            "sell_aggression_ratio",
            self.aggression_metrics.sell_aggression_ratio,
        );

        // Arrival rate metrics
        features.add("arrival_rate_buy", self.arrival_metrics.buy_arrival_rate);
        features.add("arrival_rate_sell", self.arrival_metrics.sell_arrival_rate);
        features.add(
            "arrival_rate_total",
            self.arrival_metrics.total_arrival_rate,
        );
        features.add(
            "inter_arrival_time_avg",
            self.arrival_metrics.avg_inter_arrival_time,
        );
        features.add(
            "arrival_rate_volatility",
            self.arrival_metrics.arrival_rate_volatility,
        );

        // Size distribution metrics
        features.add(
            "trade_size_small_volume",
            self.size_metrics.small_trade_volume,
        );
        features.add(
            "trade_size_medium_volume",
            self.size_metrics.medium_trade_volume,
        );
        features.add(
            "trade_size_large_volume",
            self.size_metrics.large_trade_volume,
        );
        features.add("trade_size_max", self.size_metrics.max_trade_size);
        features.add("trade_size_std", self.size_metrics.trade_size_std);
    }
}

impl Default for FlowFeatures {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_imbalance() {
        let mut flow = FlowFeatures::with_window(10_000_000); // 10 seconds

        // Add some buy trades
        flow.update_trade(Price::from(100i64), Quantity::from(10u32), true, 1000);
        flow.update_trade(Price::from(101i64), Quantity::from(20u32), true, 2000);

        // Add some sell trades
        flow.update_trade(Price::from(99i64), Quantity::from(5u32), false, 3000);
        flow.update_trade(Price::from(98i64), Quantity::from(5u32), false, 4000);

        // Buy volume: 30, Sell volume: 10
        // Imbalance: (30 - 10) / (30 + 10) = 0.5
        assert_eq!(flow.flow_metrics.buy_volume, 30.0);
        assert_eq!(flow.flow_metrics.sell_volume, 10.0);
        assert_eq!(flow.flow_metrics.flow_imbalance, 0.5);
        assert_eq!(flow.flow_metrics.net_flow, 20.0);
    }

    #[test]
    fn test_arrival_rates() {
        let mut flow = FlowFeatures::with_window(10_000_000); // 10 seconds

        // Add trades over 1 second
        flow.update_trade(Price::from(100i64), Quantity::from(10u32), true, 0);
        flow.update_trade(Price::from(100i64), Quantity::from(10u32), true, 250_000);
        flow.update_trade(Price::from(100i64), Quantity::from(10u32), false, 500_000);
        flow.update_trade(Price::from(100i64), Quantity::from(10u32), true, 750_000);
        flow.update_trade(Price::from(100i64), Quantity::from(10u32), false, 1_000_000);

        // 5 trades in 1 second = 5 trades/sec
        assert_eq!(flow.arrival_metrics.total_arrival_rate, 5.0);
        // 3 buys in 1 second = 3 buys/sec
        assert_eq!(flow.arrival_metrics.buy_arrival_rate, 3.0);
        // 2 sells in 1 second = 2 sells/sec
        assert_eq!(flow.arrival_metrics.sell_arrival_rate, 2.0);
    }

    #[test]
    fn test_window_removal() {
        let mut flow = FlowFeatures::with_window(1_000_000); // 1 second window

        // Add old trade
        flow.update_trade(Price::from(100i64), Quantity::from(10u32), true, 0);

        // Add new trade after window
        flow.update_trade(Price::from(100i64), Quantity::from(20u32), true, 2_000_000);

        // Old trade should be removed
        assert_eq!(flow.trades.len(), 1);
        assert_eq!(flow.flow_metrics.buy_volume, 20.0);
    }
}
