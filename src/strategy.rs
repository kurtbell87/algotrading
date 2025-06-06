use crate::lob::Market;
use databento::dbn::{enums::Action, record::MboMsg};
use std::collections::VecDeque;

/// Possible actions returned by [`HelloStrategy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeAction {
    /// Enter a long position.
    Long,
    /// Enter a short position.
    Short,
    /// Exit any open position.
    Flatten,
    /// Do nothing.
    Hold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Position {
    Flat,
    Long,
    Short,
}

/// A minimal example trading strategy.
///
/// The strategy keeps a rolling mean of the last `window` trade prices. On
/// every trade message it compares the current mid price to this mean and
/// generates `TradeAction`s to go long, short, flatten, or hold.
#[derive(Debug)]
pub struct HelloStrategy {
    market: Market,
    window: usize,
    trades: VecDeque<f64>,
    position: Position,
}

impl HelloStrategy {
    /// Create a new strategy with the specified rolling window size.
    pub fn new(window: usize) -> Self {
        Self {
            market: Market::default(),
            window,
            trades: VecDeque::with_capacity(window),
            position: Position::Flat,
        }
    }

    fn rolling_mean(&self) -> Option<f64> {
        if self.trades.is_empty() {
            None
        } else {
            Some(self.trades.iter().sum::<f64>() / self.trades.len() as f64)
        }
    }

    /// Process an incoming [`MboMsg`].
    ///
    /// Non-trade messages update the internal book state and return `Hold`.
    /// When a trade message is encountered, the book is updated and the current
    /// mid price is compared to the rolling mean of recent trades to produce a
    /// [`TradeAction`].
    pub fn on_message(&mut self, msg: MboMsg) -> TradeAction {
        let inst = msg.hd.instrument_id;

        // Always update the internal book state, even for snapshot messages
        self.market.apply(msg.clone());

        // Skip trade logic on snapshot messages
        if msg.flags.is_snapshot() {
            return TradeAction::Hold;
        }

        if msg.action().unwrap_or(Action::None) != Action::Trade {
            return TradeAction::Hold;
        }

        // update rolling trade history
        let price = msg.price as f64;
        self.trades.push_back(price);
        if self.trades.len() > self.window {
            self.trades.pop_front();
        }
        let mean = match self.rolling_mean() {
            Some(m) => m,
            None => return TradeAction::Hold,
        };

        let (bid, ask) = self.market.aggregated_bbo(inst);
        let (Some(b), Some(a)) = (bid, ask) else {
            return TradeAction::Hold;
        };
        let mid = (b.price + a.price) as f64 / 2.0;

        match self.position {
            Position::Flat => {
                if mid > mean {
                    self.position = Position::Long;
                    TradeAction::Long
                } else if mid < mean {
                    self.position = Position::Short;
                    TradeAction::Short
                } else {
                    TradeAction::Hold
                }
            }
            Position::Long => {
                if mid < mean {
                    self.position = Position::Flat;
                    TradeAction::Flatten
                } else {
                    TradeAction::Hold
                }
            }
            Position::Short => {
                if mid > mean {
                    self.position = Position::Flat;
                    TradeAction::Flatten
                } else {
                    TradeAction::Hold
                }
            }
        }
    }
}
