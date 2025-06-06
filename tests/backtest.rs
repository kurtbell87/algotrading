use algotrading::backtest::{Backtester, Signal};

#[test]
fn profit_after_round_trip() {
    let prices = [10.0, 12.0, 11.0];
    let signals = [Signal::Buy, Signal::Hold, Signal::Sell];
    let mut bt = Backtester::new();
    bt.run(&prices, &signals);
    let pnl = bt.final_profit(prices[2]);
    assert!((pnl - 1.0).abs() < 1e-6);
}
use algotrading::lob::{Market, InstId};
use databento::dbn::{record::{MboMsg, RecordHeader}, enums::{Action, Side}, FlagSet};

fn msg(order_id: u64, side: Side, action: Action, px: i64, sz: u32, inst: InstId) -> MboMsg {
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
    m.hd.publisher_id = 1;
    m
}

#[test]
fn step_mbo_isolated() {
    let mut market = Market::default();
    market.apply(msg(1, Side::Bid, Action::Add, 100, 1, 1));
    market.apply(msg(2, Side::Ask, Action::Add, 110, 1, 1));
    market.apply(msg(3, Side::Bid, Action::Add, 200, 1, 2));
    market.apply(msg(4, Side::Ask, Action::Add, 210, 1, 2));

    let mut bt1 = Backtester::new();
    let mut bt2 = Backtester::new();

    let _ = bt1.step_mbo(&market, 1, Signal::Buy).unwrap();
    let pnl1 = bt1.final_profit(105.0);
    let _ = bt2.step_mbo(&market, 2, Signal::Buy).unwrap();

    // backtester for instrument 1 unaffected by instrument 2
    assert!((bt1.final_profit(105.0) - pnl1).abs() < 1e-6);
    assert!(bt2.final_profit(205.0).abs() < 1e-6);
}

