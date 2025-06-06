//! Performance metrics for evaluating trading strategies.
//!
//! This module provides common statistics like profit and loss (PnL),
//! the Sharpe ratio and the Sortino ratio.

/// Compute the cumulative profit and loss from a series of per-period returns.
///
/// # Examples
///
/// ```
/// use algotrading::metrics::total_pnl;
/// let pnl = total_pnl(&[1.0, -0.5, 0.2]);
/// assert!((pnl - 0.7).abs() < 1e-12);
/// ```
pub fn total_pnl(returns: &[f64]) -> f64 {
    returns.iter().sum()
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn stddev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let m = mean(data);
    let var = data.iter().map(|r| (r - m).powi(2)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}

/// Compute the Sharpe ratio of a series of returns.
///
/// The ratio is the mean return divided by the standard deviation of returns.
/// Returns `None` if the standard deviation is zero or no data is provided.
pub fn sharpe_ratio(returns: &[f64]) -> Option<f64> {
    let sd = stddev(returns);
    if sd == 0.0 {
        return None;
    }
    Some(mean(returns) / sd)
}

/// Compute the Sortino ratio of a series of returns.
///
/// The denominator uses the standard deviation of only the downside returns.
/// Returns `None` if there is no downside deviation or no data.
pub fn sortino_ratio(returns: &[f64]) -> Option<f64> {
    let downside: Vec<f64> = returns.iter().copied().filter(|r| *r < 0.0).collect();
    let sd = stddev(&downside);
    if sd == 0.0 {
        return None;
    }
    Some(mean(returns) / sd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pnl_sum() {
        let pnl = total_pnl(&[1.0, -0.5, 0.2]);
        assert!((pnl - 0.7).abs() < 1e-12);
    }

    #[test]
    fn sharpe_nonzero() {
        let s = sharpe_ratio(&[1.0, 2.0, 3.0]).unwrap();
        assert!(s > 0.0);
    }

    #[test]
    fn sortino_downside_zero() {
        assert!(sortino_ratio(&[1.0, 2.0, 3.0]).is_none());
    }
}
