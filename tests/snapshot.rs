use algotrading::lob::Book;
use databento::dbn::{record::{MboMsg, RecordHeader}, enums::{Action, Side}, FlagSet};

fn msg(order_id: u64, side: Side, action: Action, px: i64, sz: u32, snapshot: bool) -> MboMsg {
    let mut flags = FlagSet::empty();
    if snapshot { flags.set_snapshot(); }
    MboMsg {
        hd: RecordHeader::new::<MboMsg>(0, 0, 1, 0),
        order_id,
        price: px,
        size: sz,
        flags,
        channel_id: 0,
        action: Into::<u8>::into(action) as i8,
        side: Into::<u8>::into(side) as i8,
        ts_recv: 0,
        ts_in_delta: 0,
        sequence: 0,
    }
}

#[test]
fn snapshot_clear_then_add() {
    let mut book = Book::default();

    book.apply(msg(1, Side::Bid, Action::Add, 100, 10, false));
    book.apply(msg(0, Side::Bid, Action::Clear, 0, 0, true));
    book.apply(msg(2, Side::Bid, Action::Add, 110, 5, true));

    let (bid, ask) = book.bbo();
    assert_eq!(bid.unwrap().price, 110);
    assert!(ask.is_none());
}
