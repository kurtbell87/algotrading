use algotrading::lob::{Book, Market};
use databento::dbn::{
    FlagSet,
    enums::{Action, Side},
    record::{MboMsg, RecordHeader},
};

fn msg(order_id: u64, side: Side, action: Action, px: i64, sz: u32) -> MboMsg {
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
fn add_and_bbo() {
    let mut book = Book::default();
    book.apply(msg(1, Side::Bid, Action::Add, 100, 5));
    book.apply(msg(2, Side::Ask, Action::Add, 110, 7));

    let (bid, ask) = book.bbo();
    assert_eq!(bid.unwrap().price, 100);
    assert_eq!(ask.unwrap().price, 110);
}

#[test]
fn modify_same_price_increase_size() {
    let mut book = Book::default();
    book.apply(msg(1, Side::Bid, Action::Add, 100, 5));
    let mut m = msg(1, Side::Bid, Action::Modify, 100, 8);
    book.apply(m);

    let (bid, ask) = book.bbo();
    assert_eq!(bid.unwrap().size, 8);
    assert!(ask.is_none());
}

#[test]
fn modify_change_price() {
    let mut book = Book::default();
    book.apply(msg(1, Side::Bid, Action::Add, 100, 5));
    book.apply(msg(1, Side::Bid, Action::Modify, 90, 5));

    let (bid, _) = book.bbo();
    assert_eq!(bid.unwrap().price, 90);
}

#[test]
fn cancel_partial_and_full() {
    let mut book = Book::default();
    book.apply(msg(1, Side::Bid, Action::Add, 100, 10));
    book.apply(msg(1, Side::Bid, Action::Cancel, 100, 4));

    let (bid, _) = book.bbo();
    assert_eq!(bid.as_ref().unwrap().size, 6);

    book.apply(msg(1, Side::Bid, Action::Cancel, 100, 6));
    assert!(book.bbo().0.is_none());
}

#[test]
fn top_of_book_clears() {
    use databento::dbn::enums::flags;

    let mut book = Book::default();
    book.apply(msg(1, Side::Bid, Action::Add, 100, 5));

    let mut flags = FlagSet::empty();
    flags.set_tob();
    let mut tob_msg = msg(0, Side::Bid, Action::Add, 101, 3);
    tob_msg.flags = flags;
    book.apply(tob_msg);

    assert_eq!(book.bids.len(), 1);
    assert_eq!(book.bbo().0.unwrap().price, 101);
}

#[test]
fn market_aggregated_bbo() {
    let mut market = Market::default();
    let mut m1 = msg(1, Side::Bid, Action::Add, 100, 5);
    m1.hd.instrument_id = 1;
    m1.hd.publisher_id = 1;
    market.apply(m1);

    let mut m2 = msg(2, Side::Bid, Action::Add, 102, 4);
    m2.hd.instrument_id = 1;
    m2.hd.publisher_id = 2;
    market.apply(m2);

    let (bb, _) = market.aggregated_bbo(1);
    assert_eq!(bb.unwrap().price, 102);
}
