use databento::dbn::{
    UNDEF_PRICE,
    enums::{Action, Side},
    pretty,
    record::MboMsg,
};
use hashbrown::HashMap;
use smallvec::{SmallVec, smallvec};
use std::collections::BTreeMap;
use std::fmt::Display;

/* ---------- fast container aliases ---------------------------------- */
type OrderMap = HashMap<u64, (Side, i64)>;
type Level = SmallVec<[MboMsg; 8]>;
type PriceLevels = BTreeMap<i64, Level>;

pub type Publisher = u8;
pub type InstId = u32;

/* ---------- helper -------------------------------------------------- */
fn agg(lvl: &Level) -> (u32, u32) {
    lvl.iter()
        .filter(|m| !m.flags.is_tob())
        .fold((0, 0), |acc, m| (acc.0 + m.size, acc.1 + 1))
}

/* ==================================================================== */
/*  BOOK – one instrument *and* one publisher                            */
/* ==================================================================== */
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelSummary {
    pub price: i64,
    pub size: u32,
    pub count: u32,
}

impl Display for LevelSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:4} @ {:6.2} | {:2} order(s)",
            self.size,
            pretty::Px(self.price),
            self.count
        )
    }
}

/// Order book for a single instrument and a single publisher.
///
/// The book keeps `orders_by_id` in sync with the individual price
/// levels stored in `bids` and `asks`. Each order appears exactly once
/// in the level corresponding to its price and side.
#[derive(Default, Debug)]
pub struct Book {
    pub orders_by_id: OrderMap,
    pub bids: PriceLevels,
    pub asks: PriceLevels,
}

impl Book {
    pub fn bbo(&self) -> (Option<LevelSummary>, Option<LevelSummary>) {
        let bid = self.bids.iter().next_back().map(|(px, l)| {
            let (sz, ct) = agg(l);
            LevelSummary {
                price: *px,
                size: sz,
                count: ct,
            }
        });
        let ask = self.asks.iter().next().map(|(px, l)| {
            let (sz, ct) = agg(l);
            LevelSummary {
                price: *px,
                size: sz,
                count: ct,
            }
        });
        (bid, ask)
    }

    pub fn apply(&mut self, mbo: MboMsg) {
        match mbo.action().unwrap() {
            Action::Add => self.add(mbo),
            Action::Modify => self.modify(mbo),
            Action::Cancel => self.cancel(mbo),
            Action::Clear => self.clear(),
            _ => {}
        }
    }

    /* -------------- internal helpers -------------------------------- */
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

    /* -------------- ADD --------------------------------------------- */
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

    /* -------------- CANCEL ------------------------------------------ */
    fn cancel(&mut self, mbo: MboMsg) {
        let (side, px) = match self.orders_by_id.get(&mbo.order_id).copied() {
            Some(v) => v,
            None => return,
        };
        let mut remove_mapping = false;
        if let Some(level) = self.side_levels_mut(side).get_mut(&px) {
            if let Some(pos) = level.iter().position(|o| o.order_id == mbo.order_id) {
                let existing = &mut level[pos];
                if mbo.size >= existing.size {
                    level.remove(pos);
                    remove_mapping = true;
                } else {
                    existing.size -= mbo.size;
                }
            }
            if level.is_empty() {
                self.side_levels_mut(side).remove(&px);
            }
        }
        if remove_mapping {
            self.orders_by_id.remove(&mbo.order_id);
        }
    }

    /* -------------- MODIFY ------------------------------------------ */
    fn modify(&mut self, mbo: MboMsg) {
        let (old_side, old_px) = match self.orders_by_id.get(&mbo.order_id) {
            Some(v) => *v,
            None => return self.add(mbo),
        };
        let new_side = mbo.side().unwrap();
        let new_px = mbo.price;

        if let Some(level) = self.side_levels_mut(old_side).get_mut(&old_px) {
            if let Some(idx) = level.iter().position(|o| o.order_id == mbo.order_id) {
                let keep_priority = old_px == new_px && mbo.size >= level[idx].size;
                level[idx] = mbo.clone();
                if keep_priority {
                    self.orders_by_id.insert(mbo.order_id, (new_side, new_px));
                    return;
                }

                let order = level.remove(idx);
                if level.is_empty() {
                    self.side_levels_mut(old_side).remove(&old_px);
                }
                self.orders_by_id.insert(mbo.order_id, (new_side, new_px));
                self.side_levels_mut(new_side)
                    .entry(new_px)
                    .or_default()
                    .push(order);
                return;
            }
        }

        self.add(mbo);
    }
}

/* ==================================================================== */
/*  MARKET – instrument → Vec< (publisher , Book) >                      */
/* ==================================================================== */
/// Collection of [`Book`]s indexed by instrument identifier.
///
/// Each entry holds the set of books for every publisher that has sent
/// updates for the instrument. The inner `Vec` is never empty for any
/// key present in `books`.
#[derive(Default, Debug)]
pub struct Market {
    books: HashMap<InstId, Vec<(Publisher, Book)>>,
}

impl Market {
    pub fn apply(&mut self, mbo: MboMsg) {
        let inst = mbo.hd.instrument_id;
        let pub_id: u8 = mbo.hd.publisher_id as u8;
        let entry = self.books.entry(inst).or_default();
        let book = if let Some((_, b)) = entry.iter_mut().find(|(p, _)| *p == pub_id) {
            b
        } else {
            entry.push((pub_id, Book::default()));
            &mut entry.last_mut().unwrap().1
        };
        book.apply(mbo);
    }

    /// aggregated BBO per publisher → highest bid / lowest ask across pubs
    pub fn aggregated_bbo(&self, inst: InstId) -> (Option<LevelSummary>, Option<LevelSummary>) {
        let Some(list) = self.books.get(&inst) else {
            return (None, None);
        };
        let mut best_bid: Option<LevelSummary> = None;
        let mut best_ask: Option<LevelSummary> = None;

        for (_, book) in list {
            let (bid, ask) = book.bbo();
            if let Some(b) = bid {
                match &mut best_bid {
                    None => best_bid = Some(b),
                    Some(bb) if b.price > bb.price => best_bid = Some(b),
                    Some(bb) if b.price == bb.price => {
                        bb.size += b.size;
                        bb.count += b.count;
                    }
                    _ => {}
                }
            }
            if let Some(a) = ask {
                match &mut best_ask {
                    None => best_ask = Some(a),
                    Some(aa) if a.price < aa.price => best_ask = Some(a),
                    Some(aa) if a.price == aa.price => {
                        aa.size += a.size;
                        aa.count += a.count;
                    }
                    _ => {}
                }
            }
        }
        (best_bid, best_ask)
    }

    /// Aggregated depth across publishers for the given side. Levels are
    /// combined by price and returned in book order (bids descending, asks ascending).
    #[allow(dead_code)]
    pub fn aggregated_depth(&self, inst: InstId, side: Side, levels: usize) -> Vec<LevelSummary> {
        use std::collections::BTreeMap;
        let Some(list) = self.books.get(&inst) else {
            return Vec::new();
        };
        let mut map: BTreeMap<i64, LevelSummary> = BTreeMap::new();
        for (_, book) in list {
            let it: Box<dyn Iterator<Item = (&i64, &SmallVec<[MboMsg; 8]>)>> = match side {
                Side::Bid => Box::new(book.bids.iter().rev()),
                Side::Ask => Box::new(book.asks.iter()),
                Side::None => Box::new(std::iter::empty()),
            };
            for (px, lvl) in it.take(levels) {
                let (sz, ct) = lvl
                    .iter()
                    .filter(|m| !m.flags.is_tob())
                    .fold((0, 0), |acc, m| (acc.0 + m.size, acc.1 + 1));
                let entry = map.entry(*px).or_insert(LevelSummary {
                    price: *px,
                    size: 0,
                    count: 0,
                });
                entry.size += sz;
                entry.count += ct;
            }
        }
        let it: Box<dyn Iterator<Item = (&i64, &LevelSummary)>> = match side {
            Side::Bid => Box::new(map.iter().rev()),
            Side::Ask => Box::new(map.iter()),
            Side::None => Box::new(std::iter::empty()),
        };
        it.take(levels).map(|(_, l)| l.clone()).collect()
    }
}
