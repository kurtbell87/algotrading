use algotrading::metrics::{sharpe_ratio, sortino_ratio, total_pnl};

#[test]
fn total_pnl_accumulates() {
    let pnl = total_pnl(&[1.0, -0.5, 0.2]);
    assert!((pnl - 0.7).abs() < 1e-12);
}

#[test]
fn sharpe_positive() {
    let s = sharpe_ratio(&[0.1, 0.2, -0.05]).unwrap();
    assert!(s > 0.0);
}

#[test]
fn sortino_none_when_no_losses() {
    assert!(sortino_ratio(&[0.1, 0.2, 0.05]).is_none());
}
