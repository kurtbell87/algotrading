use crate::core::{InstrumentId, PublisherId};
use crate::order_book::book::{Book, LevelSummary};
use databento::dbn::record::MboMsg;
use hashbrown::HashMap;

/// Market structure that manages multiple books across instruments and publishers
#[derive(Default, Debug)]
pub struct Market {
    books: HashMap<InstrumentId, Vec<(PublisherId, Book)>>,
}

impl Market {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply an MBO message to the appropriate book
    pub fn apply(&mut self, mbo: MboMsg) {
        let inst = mbo.hd.instrument_id;
        let pub_id = mbo.hd.publisher_id as u8;
        let entry = self.books.entry(inst).or_default();

        let book = if let Some((_, b)) = entry.iter_mut().find(|(p, _)| *p == pub_id) {
            b
        } else {
            entry.push((pub_id, Book::new()));
            &mut entry.last_mut().unwrap().1
        };

        book.apply(mbo);
    }

    /// Get aggregated BBO across all publishers for an instrument
    pub fn aggregated_bbo(
        &self,
        inst: InstrumentId,
    ) -> (Option<LevelSummary>, Option<LevelSummary>) {
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

    /// Get a specific book for an instrument and publisher
    pub fn get_book(&self, inst: InstrumentId, pub_id: PublisherId) -> Option<&Book> {
        self.books
            .get(&inst)?
            .iter()
            .find(|(p, _)| *p == pub_id)
            .map(|(_, book)| book)
    }

    /// Get all books for an instrument
    pub fn get_books(&self, inst: InstrumentId) -> Option<&[(PublisherId, Book)]> {
        self.books.get(&inst).map(|v| v.as_slice())
    }

    /// Get all tracked instruments
    pub fn instruments(&self) -> Vec<InstrumentId> {
        self.books.keys().copied().collect()
    }
}
