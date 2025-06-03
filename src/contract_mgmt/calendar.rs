use time::{Date, Month, Weekday};

/// Futures contract months (quarterly for equity index futures)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContractMonth {
    March,
    June,
    September,
    December,
}

impl ContractMonth {
    /// Get the single-letter code for CME futures
    pub fn code(&self) -> char {
        match self {
            ContractMonth::March => 'H',
            ContractMonth::June => 'M',
            ContractMonth::September => 'U',
            ContractMonth::December => 'Z',
        }
    }

    /// Convert to time::Month
    pub fn to_month(&self) -> Month {
        match self {
            ContractMonth::March => Month::March,
            ContractMonth::June => Month::June,
            ContractMonth::September => Month::September,
            ContractMonth::December => Month::December,
        }
    }

    /// Create from month code
    pub fn from_code(code: char) -> Option<Self> {
        match code.to_ascii_uppercase() {
            'H' => Some(ContractMonth::March),
            'M' => Some(ContractMonth::June),
            'U' => Some(ContractMonth::September),
            'Z' => Some(ContractMonth::December),
            _ => None,
        }
    }

    /// Get the next contract month
    pub fn next(&self) -> Self {
        match self {
            ContractMonth::March => ContractMonth::June,
            ContractMonth::June => ContractMonth::September,
            ContractMonth::September => ContractMonth::December,
            ContractMonth::December => ContractMonth::March,
        }
    }

    /// Get all contract months in order
    pub fn all() -> [Self; 4] {
        [
            ContractMonth::March,
            ContractMonth::June,
            ContractMonth::September,
            ContractMonth::December,
        ]
    }
}

/// Get the expiration date (third Friday of the month)
pub fn get_expiration_date(year: i32, month: ContractMonth) -> Date {
    get_nth_weekday_of_month(year, month.to_month(), Weekday::Friday, 3)
}

/// Get the roll date (second Thursday of the month, 8 days before expiration)
pub fn get_roll_date(year: i32, month: ContractMonth) -> Date {
    get_nth_weekday_of_month(year, month.to_month(), Weekday::Thursday, 2)
}

/// Get the Nth occurrence of a weekday in a month
fn get_nth_weekday_of_month(year: i32, month: Month, weekday: Weekday, n: u8) -> Date {
    let mut count = 0;

    for day in 1..=31 {
        if let Ok(date) = Date::from_calendar_date(year, month, day) {
            if date.weekday() == weekday {
                count += 1;
                if count == n {
                    return date;
                }
            }
        } else {
            break;
        }
    }

    panic!(
        "Could not find {} occurrence of {:?} in {}-{}",
        n, weekday, year, month
    );
}

/// Get the front month contract for a given date
pub fn get_front_month(date: Date) -> (ContractMonth, i32) {
    let year = date.year();

    // Check each quarterly month
    for month in ContractMonth::all() {
        let roll_date = get_roll_date(year, month);
        if date < roll_date {
            return (month, year);
        }
    }

    // If we're past December roll, next front month is March of next year
    (ContractMonth::March, year + 1)
}

/// Check if a date is within the roll period (8 days before expiration)
pub fn is_in_roll_period(date: Date, contract_month: ContractMonth, contract_year: i32) -> bool {
    let roll_date = get_roll_date(contract_year, contract_month);
    let expiration_date = get_expiration_date(contract_year, contract_month);

    date >= roll_date && date < expiration_date
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::date;

    #[test]
    fn test_contract_month_code() {
        assert_eq!(ContractMonth::March.code(), 'H');
        assert_eq!(ContractMonth::June.code(), 'M');
        assert_eq!(ContractMonth::September.code(), 'U');
        assert_eq!(ContractMonth::December.code(), 'Z');
    }

    #[test]
    fn test_contract_month_from_code() {
        assert_eq!(ContractMonth::from_code('H'), Some(ContractMonth::March));
        assert_eq!(ContractMonth::from_code('m'), Some(ContractMonth::June));
        assert_eq!(
            ContractMonth::from_code('U'),
            Some(ContractMonth::September)
        );
        assert_eq!(ContractMonth::from_code('z'), Some(ContractMonth::December));
        assert_eq!(ContractMonth::from_code('X'), None);
    }

    #[test]
    fn test_next_contract_month() {
        assert_eq!(ContractMonth::March.next(), ContractMonth::June);
        assert_eq!(ContractMonth::June.next(), ContractMonth::September);
        assert_eq!(ContractMonth::September.next(), ContractMonth::December);
        assert_eq!(ContractMonth::December.next(), ContractMonth::March);
    }

    #[test]
    fn test_expiration_dates() {
        // March 2025 third Friday is March 21
        assert_eq!(
            get_expiration_date(2025, ContractMonth::March),
            date!(2025 - 03 - 21)
        );

        // June 2025 third Friday is June 20
        assert_eq!(
            get_expiration_date(2025, ContractMonth::June),
            date!(2025 - 06 - 20)
        );

        // September 2025 third Friday is September 19
        assert_eq!(
            get_expiration_date(2025, ContractMonth::September),
            date!(2025 - 09 - 19)
        );

        // December 2025 third Friday is December 19
        assert_eq!(
            get_expiration_date(2025, ContractMonth::December),
            date!(2025 - 12 - 19)
        );
    }

    #[test]
    fn test_roll_dates() {
        // March 2025 second Thursday is March 13
        assert_eq!(
            get_roll_date(2025, ContractMonth::March),
            date!(2025 - 03 - 13)
        );

        // June 2025 second Thursday is June 12
        assert_eq!(
            get_roll_date(2025, ContractMonth::June),
            date!(2025 - 06 - 12)
        );

        // September 2025 second Thursday is September 11
        assert_eq!(
            get_roll_date(2025, ContractMonth::September),
            date!(2025 - 09 - 11)
        );

        // December 2025 second Thursday is December 11
        assert_eq!(
            get_roll_date(2025, ContractMonth::December),
            date!(2025 - 12 - 11)
        );
    }

    #[test]
    fn test_get_front_month() {
        // Before March roll
        assert_eq!(
            get_front_month(date!(2025 - 03 - 01)),
            (ContractMonth::March, 2025)
        );
        assert_eq!(
            get_front_month(date!(2025 - 03 - 12)),
            (ContractMonth::March, 2025)
        );

        // After March roll, before June roll
        assert_eq!(
            get_front_month(date!(2025 - 03 - 13)),
            (ContractMonth::June, 2025)
        );
        assert_eq!(
            get_front_month(date!(2025 - 05 - 01)),
            (ContractMonth::June, 2025)
        );

        // After June roll, before September roll
        assert_eq!(
            get_front_month(date!(2025 - 06 - 12)),
            (ContractMonth::September, 2025)
        );
        assert_eq!(
            get_front_month(date!(2025 - 08 - 01)),
            (ContractMonth::September, 2025)
        );

        // After September roll, before December roll
        assert_eq!(
            get_front_month(date!(2025 - 09 - 11)),
            (ContractMonth::December, 2025)
        );
        assert_eq!(
            get_front_month(date!(2025 - 11 - 01)),
            (ContractMonth::December, 2025)
        );

        // After December roll
        assert_eq!(
            get_front_month(date!(2025 - 12 - 11)),
            (ContractMonth::March, 2026)
        );
        assert_eq!(
            get_front_month(date!(2025 - 12 - 31)),
            (ContractMonth::March, 2026)
        );
    }

    #[test]
    fn test_is_in_roll_period() {
        // Test June 2025 roll period (June 12-20)
        assert!(!is_in_roll_period(
            date!(2025 - 06 - 11),
            ContractMonth::June,
            2025
        ));
        assert!(is_in_roll_period(
            date!(2025 - 06 - 12),
            ContractMonth::June,
            2025
        ));
        assert!(is_in_roll_period(
            date!(2025 - 06 - 19),
            ContractMonth::June,
            2025
        ));
        assert!(!is_in_roll_period(
            date!(2025 - 06 - 20),
            ContractMonth::June,
            2025
        ));
    }
}
