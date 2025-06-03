use crate::core::{BookDepth, OrderBook, Price, PriceLevel, Quantity};
use databento::dbn::{
    UNDEF_PRICE,
    enums::{Action, Side},
    record::MboMsg,
};
use hashbrown::HashMap;
use smallvec::{SmallVec, smallvec};
use std::collections::BTreeMap;

// Fast container aliases
type OrderMap = HashMap<u64, (Side, i64)>;
type Level = SmallVec<[MboMsg; 8]>;
type PriceLevels = BTreeMap<i64, Level>;

/// Single instrument order book for one publisher
#[derive(Default, Debug)]
pub struct Book {
    orders_by_id: OrderMap,
    bids: PriceLevels,
    asks: PriceLevels,
}

impl Book {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply an MBO message to update the book
    pub fn apply(&mut self, mbo: MboMsg) {
        match mbo.action().unwrap() {
            Action::Add => self.add(mbo),
            Action::Modify => self.modify(mbo),
            Action::Cancel => self.cancel(mbo),
            Action::Clear => self.clear(),
            _ => {}
        }
    }

    /// Get the best bid and ask levels
    pub fn bbo(&self) -> (Option<LevelSummary>, Option<LevelSummary>) {
        let bid = self.bids.iter().rev().next().map(|(px, l)| {
            let (sz, ct) = aggregate_level(l);
            LevelSummary {
                price: *px,
                size: sz,
                count: ct,
            }
        });
        let ask = self.asks.iter().next().map(|(px, l)| {
            let (sz, ct) = aggregate_level(l);
            LevelSummary {
                price: *px,
                size: sz,
                count: ct,
            }
        });
        (bid, ask)
    }

    // Internal helpers
    fn side_levels_mut(&mut self, s: Side) -> &mut PriceLevels {
        match s {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
            Side::None => unreachable!(),
        }
    }

    fn clear(&mut self) {
        self.orders_by_id.clear();
        self.bids.clear();
        self.asks.clear();
    }

    fn add(&mut self, mbo: MboMsg) {
        let side = mbo.side().unwrap();
        let px = mbo.price;
        if mbo.flags.is_tob() {
            let levels = self.side_levels_mut(side);
            levels.clear();
            if px != UNDEF_PRICE {
                levels.insert(px, smallvec![mbo]);
            }
            return;
        }
        self.orders_by_id.insert(mbo.order_id, (side, px));
        self.side_levels_mut(side).entry(px).or_default().push(mbo);
    }

    fn cancel(&mut self, mbo: MboMsg) {
        let (side, px) = match self.orders_by_id.remove(&mbo.order_id) {
            Some(v) => v,
            None => return,
        };
        if let Some(level) = self.side_levels_mut(side).get_mut(&px) {
            if let Some(pos) = level.iter().position(|o| o.order_id == mbo.order_id) {
                let existing = &mut level[pos];
                if mbo.size >= existing.size {
                    level.remove(pos);
                } else {
                    existing.size -= mbo.size;
                }
            }
            if level.is_empty() {
                self.side_levels_mut(side).remove(&px);
            }
        }
    }

    fn modify(&mut self, mbo: MboMsg) {
        let (old_side, old_px) = match self.orders_by_id.get(&mbo.order_id) {
            Some(v) => *v,
            None => return self.add(mbo),
        };
        let new_side = mbo.side().unwrap();
        let new_px = mbo.price;
        self.orders_by_id.insert(mbo.order_id, (new_side, new_px));

        let Some(level) = self.side_levels_mut(old_side).get_mut(&old_px) else {
            return self.add(mbo);
        };
        let Some(idx) = level.iter().position(|o| o.order_id == mbo.order_id) else {
            return self.add(mbo);
        };

        let keep_priority = old_px == new_px && mbo.size >= level[idx].size;
        level[idx] = mbo.clone();
        if keep_priority {
            return;
        }

        let order = level.remove(idx);
        if level.is_empty() {
            self.side_levels_mut(old_side).remove(&old_px);
        }
        self.side_levels_mut(new_side)
            .entry(new_px)
            .or_default()
            .push(order);
    }
}

/// Implement the OrderBook trait for Book
impl OrderBook for Book {
    fn best_bid(&self) -> Option<PriceLevel> {
        self.bbo().0.map(|l| l.into())
    }

    fn best_ask(&self) -> Option<PriceLevel> {
        self.bbo().1.map(|l| l.into())
    }

    fn depth(&self, levels: usize) -> BookDepth {
        let mut bids = Vec::new();
        let mut asks = Vec::new();

        // Collect bids (reverse order for highest first)
        for (px, level) in self.bids.iter().rev().take(levels) {
            let (sz, ct) = aggregate_level(level);
            bids.push(PriceLevel {
                price: Price::new(*px),
                quantity: Quantity::new(sz),
                order_count: ct,
            });
        }

        // Collect asks
        for (px, level) in self.asks.iter().take(levels) {
            let (sz, ct) = aggregate_level(level);
            asks.push(PriceLevel {
                price: Price::new(*px),
                quantity: Quantity::new(sz),
                order_count: ct,
            });
        }

        BookDepth {
            bids,
            asks,
            timestamp: 0, // TODO: Add timestamp tracking
        }
    }

    fn spread(&self) -> Option<Price> {
        let (bid, ask) = self.bbo();
        match (bid, ask) {
            (Some(b), Some(a)) => Some(Price::new(a.price - b.price)),
            _ => None,
        }
    }
}

/// Summary of a price level
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelSummary {
    pub price: i64,
    pub size: u32,
    pub count: u32,
}

impl From<LevelSummary> for PriceLevel {
    fn from(ls: LevelSummary) -> Self {
        PriceLevel {
            price: Price::new(ls.price),
            quantity: Quantity::new(ls.size),
            order_count: ls.count,
        }
    }
}

/// Aggregate a level's total size and order count
fn aggregate_level(lvl: &Level) -> (u32, u32) {
    lvl.iter()
        .filter(|m| !m.flags.is_tob())
        .fold((0, 0), |acc, m| (acc.0 + m.size, acc.1 + 1))
}
