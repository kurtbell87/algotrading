use algotrading::strategy::{HelloStrategy, TradeAction};
use databento::dbn::{
    FlagSet,
    enums::{Action, Side},
    record::{MboMsg, RecordHeader},
};

fn msg(order_id: u64, side: Side, action: Action, px: i64, sz: u32, inst: u32) -> MboMsg {
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
fn basic_flow() {
    let mut strat = HelloStrategy::new(2);

    // Seed book state
    strat.on_message(msg(1, Side::Bid, Action::Add, 99, 1, 1));
    strat.on_message(msg(2, Side::Ask, Action::Add, 101, 1, 1));

    // First trade -> mean 101, mid 100 -> short
    let act = strat.on_message(msg(0, Side::Bid, Action::Trade, 101, 1, 1));
    assert_eq!(act, TradeAction::Short);

    // Second trade -> mean 99.5, mid 100 -> flatten
    let act = strat.on_message(msg(0, Side::Bid, Action::Trade, 98, 1, 1));
    assert_eq!(act, TradeAction::Flatten);

    // Third trade -> mean 100, mid 100 -> hold
    let act = strat.on_message(msg(0, Side::Bid, Action::Trade, 102, 1, 1));
    assert_eq!(act, TradeAction::Hold);
}

#[test]
fn ignore_snapshot_messages() {
    let mut strat = HelloStrategy::new(2);

    let mut s1 = msg(1, Side::Bid, Action::Add, 99, 1, 1);
    s1.flags.set_snapshot();
    assert_eq!(strat.on_message(s1), TradeAction::Hold);

    let mut s2 = msg(2, Side::Ask, Action::Add, 101, 1, 1);
    s2.flags.set_snapshot();
    assert_eq!(strat.on_message(s2), TradeAction::Hold);

    // subsequent live trade should trigger normally using the snapshot state
    let act = strat.on_message(msg(0, Side::Bid, Action::Trade, 101, 1, 1));
    assert_eq!(act, TradeAction::Short);
}

#[test]
fn pnl_of_example() {
    use algotrading::lob::Market;

    let mut strat = HelloStrategy::new(2);
    let mut market = Market::default();

    let mut cash = 0.0;
    let mut position: i32 = 0;

    let mut step = |m: MboMsg| {
        market.apply(m.clone());
        let act = strat.on_message(m);
        let (bid, ask) = market.aggregated_bbo(1);
        let (Some(b), Some(a)) = (bid, ask) else {
            return;
        };
        let mid = (b.price + a.price) as f64 / 2.0;
        match act {
            TradeAction::Long => {
                cash -= mid;
                position = 1;
            }
            TradeAction::Short => {
                cash += mid;
                position = -1;
            }
            TradeAction::Flatten => {
                cash += position as f64 * mid;
                position = 0;
            }
            TradeAction::Hold => {}
        }
    };

    // seed BBO
    step(msg(1, Side::Bid, Action::Add, 100, 1, 1));
    step(msg(2, Side::Ask, Action::Add, 102, 1, 1));

    // first trade opens a long position
    step(msg(0, Side::Bid, Action::Trade, 99, 1, 1));

    // move the ask higher
    step(msg(2, Side::Ask, Action::Modify, 120, 1, 1));

    // more trades update the rolling mean
    step(msg(0, Side::Bid, Action::Trade, 118, 1, 1));
    step(msg(0, Side::Bid, Action::Trade, 200, 1, 1));

    let (bid, ask) = market.aggregated_bbo(1);
    let (Some(b), Some(a)) = (bid, ask) else {
        panic!("no mid")
    };
    let final_mid = (b.price + a.price) as f64 / 2.0;
    let pnl = cash + position as f64 * final_mid;

    assert!((pnl - 9.0).abs() < 1e-6);
}
