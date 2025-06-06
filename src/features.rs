//! Feature extraction utilities for Market snapshots.
use crate::lob::{Market, Book, LevelSummary, InstId};
use databento::dbn::{enums::Side, record::MboMsg};
use smallvec::SmallVec;

/// Number of levels captured for depth-related features.
pub const DEFAULT_LEVELS: usize = 5;

/// Feature vector derived from a [`Market`] snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    pub mid_px: Option<f64>,
    pub spread: Option<f64>,
    pub bid_sizes: Vec<u32>,
    pub ask_sizes: Vec<u32>,
    pub bid_counts: Vec<u32>,
    pub ask_counts: Vec<u32>,
    pub imbalance: Option<f64>,
}

impl Default for Features {
    fn default() -> Self {
        Self {
            mid_px: None,
            spread: None,
            bid_sizes: Vec::new(),
            ask_sizes: Vec::new(),
            bid_counts: Vec::new(),
            ask_counts: Vec::new(),
            imbalance: None,
        }
    }
}

impl Book {
    /// Return summaries of the top `n` price levels on one side of the book.
    pub fn top_levels(&self, side: Side, n: usize) -> Vec<LevelSummary> {
        let iter: Box<dyn Iterator<Item = (&i64, &SmallVec<[MboMsg; 8]>)> + '_> = match side {
            Side::Bid => Box::new(self.bids.iter().rev()),
            Side::Ask => Box::new(self.asks.iter()),
            Side::None => Box::new(std::iter::empty()),
        };

        iter.take(n)
            .map(|(px, lvl)| {
                let (sz, ct) = lvl
                    .iter()
                    .filter(|m| !m.flags.is_tob())
                    .fold((0, 0), |acc, m| (acc.0 + m.size, acc.1 + 1));
                LevelSummary { price: *px, size: sz, count: ct }
            })
            .collect()
    }
}

impl Market {
    /// Extract [`Features`] for a particular instrument.
    pub fn extract_features(&self, inst: InstId, levels: usize) -> Option<Features> {
        let bids = self.aggregated_depth(inst, Side::Bid, levels);
        let asks = self.aggregated_depth(inst, Side::Ask, levels);
        if bids.is_empty() && asks.is_empty() {
            return None;
        }
        let bid_px = bids.get(0).map(|b| b.price as f64);
        let ask_px = asks.get(0).map(|a| a.price as f64);
        let mid_px = match (bid_px, ask_px) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        };
        let spread = match (bid_px, ask_px) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        };
        let imbalance = match (bids.get(0), asks.get(0)) {
            (Some(b), Some(a)) if b.size + a.size > 0 => {
                Some((b.size as f64 - a.size as f64) / (b.size + a.size) as f64)
            }
            _ => None,
        };
        let mut feat = Features {
            mid_px,
            spread,
            bid_sizes: Vec::with_capacity(levels),
            ask_sizes: Vec::with_capacity(levels),
            bid_counts: Vec::with_capacity(levels),
            ask_counts: Vec::with_capacity(levels),
            imbalance,
        };
        feat.bid_sizes.extend(bids.iter().map(|l| l.size));
        feat.ask_sizes.extend(asks.iter().map(|l| l.size));
        feat.bid_counts.extend(bids.iter().map(|l| l.count));
        feat.ask_counts.extend(asks.iter().map(|l| l.count));
        Some(feat)
    }
}

#[cfg(test)]
mod tests {
    use crate::lob::Market;
    use databento::dbn::{record::MboMsg, RecordHeader, enums::{Action, Side}, FlagSet};

    fn msg(order_id: u64, side: Side, action: Action, px: i64, sz: u32, _pub_id: u8) -> MboMsg {
        MboMsg {
            hd: RecordHeader::new::<MboMsg>(0, 0, 1, 0),
            order_id,
            price: px,
            size: sz,
            flags: FlagSet::empty(),
            channel_id: 0,
            action: Into::<u8>::into(action) as i8,
            side: Into::<u8>::into(side) as i8,
            ts_recv: 0,
            ts_in_delta: 0,
            sequence: 0,
        }
    }

    #[test]
    fn extract_mid_spread_depth() {
        let mut market = Market::default();
        let mut m1 = msg(1, Side::Bid, Action::Add, 100, 5, 1);
        m1.hd.instrument_id = 1;
        m1.hd.publisher_id = 1;
        market.apply(m1);
        let mut m2 = msg(2, Side::Ask, Action::Add, 110, 7, 1);
        m2.hd.instrument_id = 1;
        m2.hd.publisher_id = 1;
        market.apply(m2);

        let feats = market.extract_features(1, 5).unwrap();
        assert_eq!(feats.mid_px.unwrap(), 105.0);
        assert_eq!(feats.spread.unwrap(), 10.0);
        assert_eq!(feats.bid_sizes[0], 5);
        assert_eq!(feats.ask_sizes[0], 7);
        assert_eq!(feats.bid_counts[0], 1);
        assert_eq!(feats.ask_counts[0], 1);
        let expected = ((5.0_f64 - 7.0_f64) / 12.0_f64).abs();
        assert!((feats.imbalance.unwrap().abs() - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregated_depth_across_publishers() {
        let mut market = Market::default();
        let mut m1 = msg(1, Side::Bid, Action::Add, 100, 5, 1);
        m1.hd.instrument_id = 1;
        m1.hd.publisher_id = 1;
        market.apply(m1);
        let mut m2 = msg(2, Side::Bid, Action::Add, 100, 3, 2);
        m2.hd.instrument_id = 1;
        m2.hd.publisher_id = 2;
        market.apply(m2);
        let feats = market.extract_features(1, 5).unwrap();
        assert_eq!(feats.bid_sizes[0], 8); // aggregated size
        assert_eq!(feats.bid_counts[0], 2);
    }

    #[test]
    fn imbalance_none_with_single_side() {
        let mut market = Market::default();
        let mut m1 = msg(1, Side::Bid, Action::Add, 100, 5, 1);
        m1.hd.instrument_id = 2;
        m1.hd.publisher_id = 1;
        market.apply(m1);

        let feats = market.extract_features(2, 5).unwrap();
        assert!(feats.imbalance.is_none());
    }

    #[test]
    fn book_top_levels_order() {
        use crate::lob::Book;

        let mut book = Book::default();
        book.apply(msg(1, Side::Bid, Action::Add, 101, 4, 0));
        book.apply(msg(2, Side::Bid, Action::Add, 100, 6, 0));

        let levels = book.top_levels(Side::Bid, 2);
        assert_eq!(levels[0].price, 101);
        assert_eq!(levels[1].price, 100);
    }

    #[test]
    fn no_features_when_no_book() {
        let market = Market::default();
        assert!(market.extract_features(1, 5).is_none());
    }
}
