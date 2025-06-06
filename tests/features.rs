use algotrading::lob::{Market, Book};
use algotrading::features::{DEFAULT_LEVELS};
use databento::dbn::{record::{MboMsg, RecordHeader}, enums::{Action, Side}, FlagSet};

fn msg(order_id: u64, side: Side, action: Action, px: i64, sz: u32, pub_id: u8, inst: u32) -> MboMsg {
    let mut m = MboMsg {
        hd: RecordHeader::new::<MboMsg>(0, 0, inst, 0),
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
    };
    m.hd.publisher_id = pub_id as u16;
    m
}

#[test]
fn no_mid_or_spread_with_single_side() {
    let mut market = Market::default();
    let m1 = msg(1, Side::Bid, Action::Add, 100, 4, 1, 1);
    market.apply(m1);

    let feats = market.extract_features(1, DEFAULT_LEVELS).unwrap();
    assert!(feats.mid_px.is_none());
    assert!(feats.spread.is_none());
}

#[test]
fn aggregated_depth_combines_asks() {
    let mut market = Market::default();
    let m1 = msg(1, Side::Ask, Action::Add, 110, 2, 1, 1);
    market.apply(m1);
    let m2 = msg(2, Side::Ask, Action::Add, 110, 4, 2, 1);
    market.apply(m2);
    let features = market.extract_features(1, DEFAULT_LEVELS).unwrap();
    assert_eq!(features.ask_sizes[0], 6);
    assert_eq!(features.ask_counts[0], 2);
}

#[test]
fn tob_orders_are_ignored_in_top_levels() {
    let mut book = Book::default();
    let mut tob = msg(1, Side::Bid, Action::Add, 101, 5, 0, 0);
    tob.flags.set_tob();
    book.apply(tob);
    book.apply(msg(2, Side::Bid, Action::Add, 101, 3, 0, 0));
    let levels = book.top_levels(Side::Bid, 1);
    assert_eq!(levels[0].size, 3);
    assert_eq!(levels[0].count, 1);
}

#[test]
fn depth_sorted_by_price() {
    let mut market = Market::default();
    let b1 = msg(1, Side::Bid, Action::Add, 100, 1, 1, 1);
    market.apply(b1);
    let b2 = msg(2, Side::Bid, Action::Add, 99, 1, 1, 1);
    market.apply(b2);
    let b3 = msg(3, Side::Bid, Action::Add, 101, 1, 1, 1);
    market.apply(b3);

    let depth = market.aggregated_depth(1, Side::Bid, 3);
    assert_eq!(depth[0].price, 101);
    assert_eq!(depth[1].price, 100);
    assert_eq!(depth[2].price, 99);
}
