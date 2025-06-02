use crate::core::InstrumentId;
use crate::contract_mgmt::calendar::{ContractMonth, get_front_month, get_roll_date, is_in_roll_period};
use std::collections::HashMap;
use time::Date;

/// Represents a futures contract
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuturesContract {
    pub symbol: String,
    pub root: String,  // e.g., "MES"
    pub month: ContractMonth,
    pub year: i32,
    pub instrument_id: InstrumentId,
}

impl FuturesContract {
    /// Create a new futures contract
    pub fn new(symbol: String, root: String, month: ContractMonth, year: i32, instrument_id: InstrumentId) -> Self {
        Self { symbol, root, month, year, instrument_id }
    }
    
    /// Get the contract code (e.g., "MESM5" for June 2025)
    pub fn code(&self) -> String {
        let year_digit = self.year % 10;
        format!("{}{}{}", self.root, self.month.code(), year_digit)
    }
    
    /// Check if this contract has expired as of the given date
    pub fn is_expired(&self, date: Date) -> bool {
        let expiration = crate::contract_mgmt::calendar::get_expiration_date(self.year, self.month);
        date >= expiration
    }
    
    /// Check if this contract is in its roll period
    pub fn is_rolling(&self, date: Date) -> bool {
        is_in_roll_period(date, self.month, self.year)
    }
}

/// Contract selector that manages contract selection based on calendar rules
#[derive(Debug, Default)]
pub struct ContractSelector {
    /// All available contracts by root symbol
    contracts_by_root: HashMap<String, Vec<FuturesContract>>,
    /// Quick lookup by instrument ID
    contracts_by_id: HashMap<InstrumentId, FuturesContract>,
}

impl ContractSelector {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a contract to the selector
    pub fn add_contract(&mut self, contract: FuturesContract) {
        self.contracts_by_id.insert(contract.instrument_id, contract.clone());
        self.contracts_by_root
            .entry(contract.root.clone())
            .or_default()
            .push(contract);
    }
    
    /// Load contracts from Databento symbology
    pub fn load_from_symbology(&mut self, mappings: &[databento::dbn::SymbolMapping]) {
        for mapping in mappings {
            if let Some(contract) = parse_futures_symbol(&mapping.raw_symbol) {
                if let Ok(inst_id) = mapping.intervals[0].symbol.parse::<u32>() {
                    let futures_contract = FuturesContract {
                        symbol: mapping.raw_symbol.clone(),
                        root: contract.0,
                        month: contract.1,
                        year: contract.2,
                        instrument_id: inst_id,
                    };
                    self.add_contract(futures_contract);
                }
            }
        }
        
        // Sort contracts by expiration for each root
        for contracts in self.contracts_by_root.values_mut() {
            contracts.sort_by_key(|c| (c.year, c.month));
        }
    }
    
    /// Get the active contract for a given root symbol and date
    pub fn get_active_contract(&self, root: &str, date: Date) -> Option<&FuturesContract> {
        let contracts = self.contracts_by_root.get(root)?;
        let (target_month, target_year) = get_front_month(date);
        
        // Find the contract matching our target
        contracts.iter()
            .find(|c| c.month == target_month && c.year == target_year)
    }
    
    /// Get the next contract to roll into
    pub fn get_next_contract(&self, current: &FuturesContract, date: Date) -> Option<&FuturesContract> {
        let contracts = self.contracts_by_root.get(&current.root)?;
        
        // If we're in roll period, get the next contract
        if current.is_rolling(date) {
            let current_idx = contracts.iter().position(|c| c == current)?;
            contracts.get(current_idx + 1)
        } else {
            None
        }
    }
    
    /// Get a contract by instrument ID
    pub fn get_contract_by_id(&self, instrument_id: InstrumentId) -> Option<&FuturesContract> {
        self.contracts_by_id.get(&instrument_id)
    }
    
    /// Get all contracts for a root symbol
    pub fn get_contracts_for_root(&self, root: &str) -> Option<&[FuturesContract]> {
        self.contracts_by_root.get(root).map(|v| v.as_slice())
    }
    
    /// Get the contract chain for backtesting (all contracts needed for a date range)
    pub fn get_contract_chain(&self, root: &str, start_date: Date, end_date: Date) -> Vec<&FuturesContract> {
        let mut chain = Vec::new();
        let mut current_date = start_date;
        
        while current_date <= end_date {
            if let Some(contract) = self.get_active_contract(root, current_date) {
                if !chain.iter().any(|&c| c == contract) {
                    chain.push(contract);
                }
                // Jump to next roll date
                let next_roll = get_roll_date(contract.year, contract.month);
                current_date = next_roll.max(current_date);
            }
            
            // Move forward by a day if we couldn't find a contract or to continue searching
            current_date = current_date.next_day().unwrap_or(current_date);
        }
        
        chain
    }
}

/// Parse a futures symbol like "MESM5" into (root, month, year)
fn parse_futures_symbol(symbol: &str) -> Option<(String, ContractMonth, i32)> {
    if symbol.len() < 4 {
        return None;
    }
    
    // Split into root and contract code
    let (root, code) = symbol.split_at(symbol.len() - 2);
    let mut chars = code.chars();
    
    // Parse month code
    let month_char = chars.next()?;
    let month = ContractMonth::from_code(month_char)?;
    
    // Parse year digit
    let year_digit = chars.next()?.to_digit(10)? as i32;
    let year = if year_digit <= 9 {
        2020 + year_digit
    } else {
        2010 + year_digit
    };
    
    Some((root.to_string(), month, year))
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::date;
    
    #[test]
    fn test_futures_contract() {
        let contract = FuturesContract::new(
            "MESM5".to_string(),
            "MES".to_string(),
            ContractMonth::June,
            2025,
            42003627,
        );
        
        assert_eq!(contract.code(), "MESM5");
        assert!(!contract.is_expired(date!(2025-06-19)));
        assert!(contract.is_expired(date!(2025-06-20)));
        assert!(contract.is_rolling(date!(2025-06-12)));
        assert!(!contract.is_rolling(date!(2025-06-11)));
    }
    
    #[test]
    fn test_parse_futures_symbol() {
        assert_eq!(
            parse_futures_symbol("MESM5"),
            Some(("MES".to_string(), ContractMonth::June, 2025))
        );
        assert_eq!(
            parse_futures_symbol("ESZ4"),
            Some(("ES".to_string(), ContractMonth::December, 2024))
        );
        assert_eq!(parse_futures_symbol("ABC"), None);
        assert_eq!(parse_futures_symbol(""), None);
    }
    
    #[test]
    fn test_contract_selector() {
        let mut selector = ContractSelector::new();
        
        // Add some test contracts
        selector.add_contract(FuturesContract::new(
            "MESM5".to_string(),
            "MES".to_string(),
            ContractMonth::June,
            2025,
            1001,
        ));
        selector.add_contract(FuturesContract::new(
            "MESU5".to_string(),
            "MES".to_string(),
            ContractMonth::September,
            2025,
            1002,
        ));
        selector.add_contract(FuturesContract::new(
            "MESZ5".to_string(),
            "MES".to_string(),
            ContractMonth::December,
            2025,
            1003,
        ));
        
        // Test getting active contract
        let active = selector.get_active_contract("MES", date!(2025-05-01)).unwrap();
        assert_eq!(active.month, ContractMonth::June);
        
        let active = selector.get_active_contract("MES", date!(2025-06-12)).unwrap();
        assert_eq!(active.month, ContractMonth::September);
        
        // Test getting next contract during roll
        let june = selector.get_contract_by_id(1001).unwrap();
        let next = selector.get_next_contract(june, date!(2025-06-12)).unwrap();
        assert_eq!(next.month, ContractMonth::September);
        
        // Test contract chain
        let chain = selector.get_contract_chain("MES", date!(2025-05-01), date!(2025-10-01));
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].month, ContractMonth::June);
        assert_eq!(chain[1].month, ContractMonth::September);
        assert_eq!(chain[2].month, ContractMonth::December);
    }
    
    #[test]
    fn test_load_from_symbology() {
        use databento::dbn::{SymbolMapping, MappingInterval};
        
        let mut selector = ContractSelector::new();
        
        let mappings = vec![
            SymbolMapping {
                raw_symbol: "MESM5".to_string(),
                intervals: vec![MappingInterval {
                    start_date: date!(2025-04-01),
                    end_date: date!(2025-06-30),
                    symbol: "1001".to_string(),
                }],
            },
            SymbolMapping {
                raw_symbol: "MESU5".to_string(),
                intervals: vec![MappingInterval {
                    start_date: date!(2025-07-01),
                    end_date: date!(2025-09-30),
                    symbol: "1002".to_string(),
                }],
            },
        ];
        
        selector.load_from_symbology(&mappings);
        
        assert_eq!(selector.get_contract_by_id(1001).unwrap().symbol, "MESM5");
        assert_eq!(selector.get_contract_by_id(1002).unwrap().symbol, "MESU5");
        
        let contracts = selector.get_contracts_for_root("MES").unwrap();
        assert_eq!(contracts.len(), 2);
        assert_eq!(contracts[0].month, ContractMonth::June);
        assert_eq!(contracts[1].month, ContractMonth::September);
    }
}