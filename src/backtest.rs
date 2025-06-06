#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug)]
pub struct Backtester {
    cash: f64,
    position: i32,
    equity: f64,
}

impl Backtester {
    /// Create a new backtester with zero cash and no position.
    pub fn new() -> Self {
        Self {
            cash: 0.0,
            position: 0,
            equity: 0.0,
        }
    }

    /// Advance the simulation by one step with a price and trading signal.
    /// Returns the reward (change in equity) after applying the signal.
    pub fn step(&mut self, price: f64, signal: Signal) -> f64 {
        match signal {
            Signal::Buy => {
                if self.position == 0 {
                    self.cash -= price;
                    self.position = 1;
                }
            }
            Signal::Sell => {
                if self.position == 1 {
                    self.cash += price;
                    self.position = 0;
                }
            }
            Signal::Hold => {}
        }
        let new_equity = self.cash + self.position as f64 * price;
        let reward = new_equity - self.equity;
        self.equity = new_equity;
        reward
    }

    /// Advance the simulation using the aggregated BBO from an [`Market`].
    /// Returns `None` if no quote is available for the given instrument.
    pub fn step_mbo(
        &mut self,
        market: &crate::lob::Market,
        inst: crate::lob::InstId,
        signal: Signal,
    ) -> Option<f64> {
        let (bid, ask) = market.aggregated_bbo(inst);
        let px = match (bid, ask) {
            (Some(b), Some(a)) => (b.price + a.price) as f64 / 2.0,
            _ => return None,
        };
        Some(self.step(px, signal))
    }

    /// Run a backtest over a series of prices and signals.
    /// Returns the reward at each step.
    pub fn run(&mut self, prices: &[f64], signals: &[Signal]) -> Vec<f64> {
        assert_eq!(prices.len(), signals.len());
        let mut rewards = Vec::with_capacity(prices.len());
        for (&p, &s) in prices.iter().zip(signals) {
            rewards.push(self.step(p, s));
        }
        rewards
    }

    /// Compute the final profit at the given price.
    pub fn final_profit(&self, price: f64) -> f64 {
        self.cash + self.position as f64 * price
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_buy_sell() {
        let prices = [10.0, 12.0, 11.0];
        let signals = [Signal::Buy, Signal::Hold, Signal::Sell];
        let mut bt = Backtester::new();
        let rewards = bt.run(&prices, &signals);
        assert_eq!(rewards.len(), 3);
        let profit = bt.final_profit(prices[2]);
        assert!((profit - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reward_sequence() {
        let prices = [1.0, 2.0, 3.0];
        let signals = [Signal::Hold, Signal::Buy, Signal::Hold];
        let mut bt = Backtester::new();
        let rewards = bt.run(&prices, &signals);
        assert_eq!(rewards, vec![0.0, 0.0, 1.0]);
    }
}
