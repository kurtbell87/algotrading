use crate::core::InstrumentId;
use std::collections::HashMap;
use databento::dbn::{SymbolMapping, Metadata};
use time::Date;

/// Manages symbol to instrument ID mappings
#[derive(Debug, Default)]
pub struct SymbologyManager {
    /// Symbol to instrument ID mapping by date
    symbol_to_id: HashMap<String, Vec<(Date, Date, InstrumentId)>>, // (start_date, end_date, id)
    /// Instrument ID to symbol mapping
    id_to_symbol: HashMap<InstrumentId, String>,
}

impl SymbologyManager {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Load symbology from Databento metadata
    pub fn load_from_metadata(&mut self, metadata: &Metadata) {
        for mapping in &metadata.mappings {
            self.add_mapping(mapping);
        }
    }
    
    /// Add a single symbol mapping
    pub fn add_mapping(&mut self, mapping: &SymbolMapping) {
        for interval in &mapping.intervals {
            if let Ok(inst_id) = interval.symbol.parse::<u32>() {
                // MappingInterval already uses u32 for dates in YYYYMMDD format
                self.symbol_to_id
                    .entry(mapping.raw_symbol.clone())
                    .or_default()
                    .push((interval.start_date, interval.end_date, inst_id));
                
                // Also store reverse mapping
                self.id_to_symbol.insert(inst_id, mapping.raw_symbol.clone());
            }
        }
    }
    
    /// Get instrument ID for a symbol on a specific date
    pub fn get_instrument_id(&self, symbol: &str, date: Date) -> Option<InstrumentId> {
        self.symbol_to_id.get(symbol)?
            .iter()
            .find(|(start, end, _)| date >= *start && date <= *end)
            .map(|(_, _, id)| *id)
    }
    
    /// Get symbol for an instrument ID
    pub fn get_symbol(&self, instrument_id: InstrumentId) -> Option<&str> {
        self.id_to_symbol.get(&instrument_id).map(|s| s.as_str())
    }
    
    /// Get all instrument IDs for a symbol across all dates
    pub fn get_all_instrument_ids(&self, symbol: &str) -> Vec<InstrumentId> {
        self.symbol_to_id.get(symbol)
            .map(|intervals| intervals.iter().map(|(_, _, id)| *id).collect())
            .unwrap_or_default()
    }
    
    /// Check if an instrument ID is known
    pub fn contains_instrument(&self, instrument_id: InstrumentId) -> bool {
        self.id_to_symbol.contains_key(&instrument_id)
    }
    
    /// Get all known symbols
    pub fn get_all_symbols(&self) -> Vec<&str> {
        self.symbol_to_id.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use databento::dbn::MappingInterval;
    use time::macros::date;
    
    #[test]
    fn test_symbology_manager() {
        let mut manager = SymbologyManager::new();
        
        // For tests, we'll add mappings directly
        manager.symbol_to_id.insert("MESM5".to_string(), vec![(date!(2025-04-01), date!(2025-06-30), 1001)]);
        manager.symbol_to_id.insert("MESU5".to_string(), vec![(date!(2025-07-01), date!(2025-09-30), 1002)]);
        manager.id_to_symbol.insert(1001, "MESM5".to_string());
        manager.id_to_symbol.insert(1002, "MESU5".to_string());
        
        // Test getting instrument IDs by date
        assert_eq!(manager.get_instrument_id("MESM5", date!(2025-05-15)), Some(1001));
        assert_eq!(manager.get_instrument_id("MESM5", date!(2025-07-01)), None);
        assert_eq!(manager.get_instrument_id("MESU5", date!(2025-08-01)), Some(1002));
        
        // Test reverse lookup
        assert_eq!(manager.get_symbol(1001), Some("MESM5"));
        assert_eq!(manager.get_symbol(1002), Some("MESU5"));
        assert_eq!(manager.get_symbol(9999), None);
        
        // Test getting all IDs for a symbol
        assert_eq!(manager.get_all_instrument_ids("MESM5"), vec![1001]);
        
        // Test contains
        assert!(manager.contains_instrument(1001));
        assert!(!manager.contains_instrument(9999));
        
        // Test getting all symbols
        let mut symbols = manager.get_all_symbols();
        symbols.sort();
        assert_eq!(symbols, vec!["MESM5", "MESU5"]);
    }
    
    #[test]
    fn test_multiple_intervals() {
        let mut manager = SymbologyManager::new();
        
        // Symbol that maps to different IDs over time
        let mapping = SymbolMapping {
            raw_symbol: "TEST".to_string(),
            intervals: vec![
                MappingInterval {
                    start_date: date!(2025-01-01),
                    end_date: date!(2025-01-31),
                    symbol: "1001".to_string(),
                },
                MappingInterval {
                    start_date: date!(2025-02-01),
                    end_date: date!(2025-02-28),
                    symbol: "1002".to_string(),
                },
            ],
        };
        
        manager.add_mapping(&mapping);
        
        // Test different dates
        assert_eq!(manager.get_instrument_id("TEST", date!(2025-01-15)), Some(1001));
        assert_eq!(manager.get_instrument_id("TEST", date!(2025-02-15)), Some(1002));
        assert_eq!(manager.get_instrument_id("TEST", date!(2025-03-01)), None);
        
        // Should return all IDs
        let mut ids = manager.get_all_instrument_ids("TEST");
        ids.sort();
        assert_eq!(ids, vec![1001, 1002]);
    }
}