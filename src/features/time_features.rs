//! Time and session-based features
//!
//! This module provides time-aware features for trading strategies including:
//! - Trading session timing (RTH vs ETH)
//! - Intraday patterns
//! - Contract expiry awareness
//! - Session volume percentiles

use crate::core::types::{InstrumentId, Quantity};
use crate::features::collector::FeatureVector;
use chrono::{DateTime, Datelike, Timelike, Utc, Weekday};
use std::collections::HashMap;

/// Trading session definition
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSession {
    /// Regular Trading Hours (RTH)
    RegularHours,
    /// Extended Trading Hours (ETH) - pre-market
    PreMarket,
    /// Extended Trading Hours (ETH) - after-hours
    AfterHours,
    /// Overnight session
    Overnight,
    /// Market closed
    Closed,
}

/// Session timing configuration for futures
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// RTH start time (UTC hour)
    pub rth_start_hour: u32,
    /// RTH end time (UTC hour)
    pub rth_end_hour: u32,
    /// Pre-market start (UTC hour)
    pub pre_market_start: u32,
    /// After-hours end (UTC hour)
    pub after_hours_end: u32,
    /// Days when market is open
    pub trading_days: Vec<Weekday>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        // Default CME futures hours (in UTC)
        // RTH: 14:30-21:00 UTC (9:30 AM - 4:00 PM ET)
        Self {
            rth_start_hour: 14,   // 9:30 AM ET
            rth_end_hour: 21,     // 4:00 PM ET
            pre_market_start: 12, // 7:00 AM ET
            after_hours_end: 22,  // 5:00 PM ET
            trading_days: vec![
                Weekday::Mon,
                Weekday::Tue,
                Weekday::Wed,
                Weekday::Thu,
                Weekday::Fri,
            ],
        }
    }
}

/// Volume profile tracking for session percentiles
#[derive(Debug, Clone)]
struct VolumeProfile {
    /// Hourly volume buckets (24 hours)
    hourly_volumes: [f64; 24],
    /// Total volume traded today
    daily_volume: f64,
    /// Rolling 20-day average volumes by hour
    avg_hourly_volumes: [f64; 24],
    /// Days tracked for averaging
    days_tracked: u32,
}

impl Default for VolumeProfile {
    fn default() -> Self {
        Self {
            hourly_volumes: [0.0; 24],
            daily_volume: 0.0,
            avg_hourly_volumes: [0.0; 24],
            days_tracked: 0,
        }
    }
}

/// Time-based features extractor
pub struct TimeFeatures {
    /// Session configuration
    config: SessionConfig,
    /// Volume profiles per instrument
    volume_profiles: HashMap<InstrumentId, VolumeProfile>,
    /// Contract expiry dates (instrument_id -> expiry_date)
    expiry_dates: HashMap<InstrumentId, DateTime<Utc>>,
    /// Front month contract ID
    front_month: Option<InstrumentId>,
    /// Last update timestamp
    last_update: Option<DateTime<Utc>>,
}

impl TimeFeatures {
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            volume_profiles: HashMap::new(),
            expiry_dates: HashMap::new(),
            front_month: None,
            last_update: None,
        }
    }

    /// Set contract expiry information
    pub fn set_expiry(&mut self, instrument_id: InstrumentId, expiry: DateTime<Utc>) {
        self.expiry_dates.insert(instrument_id, expiry);
    }

    /// Set front month contract
    pub fn set_front_month(&mut self, instrument_id: InstrumentId) {
        self.front_month = Some(instrument_id);
    }

    /// Update volume profile with trade
    pub fn update_volume(
        &mut self,
        instrument_id: InstrumentId,
        quantity: Quantity,
        timestamp: u64,
    ) {
        let dt = DateTime::from_timestamp_micros(timestamp as i64).unwrap();
        let hour = dt.hour() as usize;

        let profile = self
            .volume_profiles
            .entry(instrument_id)
            .or_insert_with(VolumeProfile::default);

        profile.hourly_volumes[hour] += quantity.as_f64();
        profile.daily_volume += quantity.as_f64();

        // Check for day rollover
        if let Some(last) = self.last_update {
            if dt.date_naive() != last.date_naive() {
                self.roll_day(instrument_id);
            }
        }

        self.last_update = Some(dt);
    }

    /// Roll to next day - update averages
    fn roll_day(&mut self, instrument_id: InstrumentId) {
        if let Some(profile) = self.volume_profiles.get_mut(&instrument_id) {
            // Update rolling averages
            let alpha = 0.05; // Exponential decay factor
            for hour in 0..24 {
                profile.avg_hourly_volumes[hour] = alpha * profile.hourly_volumes[hour]
                    + (1.0 - alpha) * profile.avg_hourly_volumes[hour];
            }

            // Reset daily volumes
            profile.hourly_volumes = [0.0; 24];
            profile.daily_volume = 0.0;
            profile.days_tracked += 1;
        }
    }

    /// Get current trading session
    pub fn get_session(&self, timestamp: u64) -> TradingSession {
        let dt = DateTime::from_timestamp_micros(timestamp as i64).unwrap();
        let hour = dt.hour();
        let weekday = dt.weekday();

        // Check if trading day
        if !self.config.trading_days.contains(&weekday) {
            return TradingSession::Closed;
        }

        // Check session times
        if hour >= self.config.rth_start_hour && hour < self.config.rth_end_hour {
            TradingSession::RegularHours
        } else if hour >= self.config.pre_market_start && hour < self.config.rth_start_hour {
            TradingSession::PreMarket
        } else if hour >= self.config.rth_end_hour && hour < self.config.after_hours_end {
            TradingSession::AfterHours
        } else if hour >= self.config.after_hours_end || hour < self.config.pre_market_start {
            TradingSession::Overnight
        } else {
            TradingSession::Closed
        }
    }

    /// Calculate time to session close (in seconds)
    pub fn time_to_session_close(&self, timestamp: u64) -> Option<f64> {
        let dt = DateTime::from_timestamp_micros(timestamp as i64).unwrap();
        let session = self.get_session(timestamp);

        let close_hour = match session {
            TradingSession::RegularHours => self.config.rth_end_hour,
            TradingSession::PreMarket => self.config.rth_start_hour,
            TradingSession::AfterHours => self.config.after_hours_end,
            _ => return None,
        };

        let current_seconds = dt.hour() * 3600 + dt.minute() * 60 + dt.second();
        let close_seconds = close_hour * 3600;

        if close_seconds > current_seconds {
            Some((close_seconds - current_seconds) as f64)
        } else {
            None
        }
    }

    /// Calculate time since session open (in seconds)
    pub fn time_since_session_open(&self, timestamp: u64) -> Option<f64> {
        let dt = DateTime::from_timestamp_micros(timestamp as i64).unwrap();
        let session = self.get_session(timestamp);

        let open_hour = match session {
            TradingSession::RegularHours => self.config.rth_start_hour,
            TradingSession::PreMarket => self.config.pre_market_start,
            TradingSession::AfterHours => self.config.rth_end_hour,
            _ => return None,
        };

        let current_seconds = dt.hour() * 3600 + dt.minute() * 60 + dt.second();
        let open_seconds = open_hour * 3600;

        if current_seconds >= open_seconds {
            Some((current_seconds - open_seconds) as f64)
        } else {
            None
        }
    }

    /// Calculate session volume percentile
    pub fn session_volume_percentile(
        &self,
        instrument_id: InstrumentId,
        timestamp: u64,
    ) -> Option<f64> {
        let profile = self.volume_profiles.get(&instrument_id)?;
        if profile.days_tracked < 5 {
            return None; // Not enough history
        }

        let dt = DateTime::from_timestamp_micros(timestamp as i64).unwrap();
        let hour = dt.hour() as usize;

        // Calculate expected volume up to current hour
        let mut expected_volume = 0.0;
        let mut actual_volume = 0.0;

        for h in 0..=hour {
            expected_volume += profile.avg_hourly_volumes[h];
            actual_volume += profile.hourly_volumes[h];
        }

        if expected_volume > 0.0 {
            Some((actual_volume / expected_volume) * 100.0)
        } else {
            None
        }
    }

    /// Days to contract expiry
    pub fn days_to_expiry(&self, instrument_id: InstrumentId, timestamp: u64) -> Option<i64> {
        let expiry = self.expiry_dates.get(&instrument_id)?;
        let current = DateTime::from_timestamp_micros(timestamp as i64).unwrap();

        let duration = expiry.signed_duration_since(current);
        Some(duration.num_days())
    }

    /// Check if this is the front month contract
    pub fn is_front_month(&self, instrument_id: InstrumentId) -> bool {
        self.front_month == Some(instrument_id)
    }

    /// Calculate roll activity indicator (0-1)
    /// Higher values indicate more rolling activity
    pub fn roll_activity_indicator(&self, instrument_id: InstrumentId, timestamp: u64) -> f64 {
        if let Some(days) = self.days_to_expiry(instrument_id, timestamp) {
            if days <= 0 {
                1.0 // Expired
            } else if days <= 3 {
                0.9 // Very close to expiry
            } else if days <= 7 {
                0.7 // Rolling week
            } else if days <= 14 {
                0.4 // Pre-roll period
            } else {
                0.1 // Normal trading
            }
        } else {
            0.0
        }
    }

    /// Add time features to feature vector
    pub fn add_to_vector(
        &self,
        features: &mut FeatureVector,
        instrument_id: InstrumentId,
        timestamp: u64,
    ) {
        let dt = DateTime::from_timestamp_micros(timestamp as i64).unwrap();

        // Basic time features
        features.add("hour_of_day", dt.hour() as f64);
        features.add("minute_of_hour", dt.minute() as f64);
        features.add("day_of_week", dt.weekday().num_days_from_monday() as f64);

        // Session features
        let session = self.get_session(timestamp);
        features.add(
            "is_rth",
            if session == TradingSession::RegularHours {
                1.0
            } else {
                0.0
            },
        );
        features.add(
            "is_pre_market",
            if session == TradingSession::PreMarket {
                1.0
            } else {
                0.0
            },
        );
        features.add(
            "is_after_hours",
            if session == TradingSession::AfterHours {
                1.0
            } else {
                0.0
            },
        );

        // Session timing
        if let Some(time_to_close) = self.time_to_session_close(timestamp) {
            features.add("seconds_to_session_close", time_to_close);
            features.add("minutes_to_session_close", time_to_close / 60.0);
        }

        if let Some(time_since_open) = self.time_since_session_open(timestamp) {
            features.add("seconds_since_session_open", time_since_open);
            features.add("minutes_since_session_open", time_since_open / 60.0);
        }

        // Volume profile
        if let Some(percentile) = self.session_volume_percentile(instrument_id, timestamp) {
            features.add("session_volume_percentile", percentile);
        }

        // Contract features
        if let Some(days) = self.days_to_expiry(instrument_id, timestamp) {
            features.add("days_to_expiry", days as f64);
            features.add("is_expiry_week", if days <= 7 { 1.0 } else { 0.0 });
        }

        features.add(
            "is_front_month",
            if self.is_front_month(instrument_id) {
                1.0
            } else {
                0.0
            },
        );
        features.add(
            "roll_activity",
            self.roll_activity_indicator(instrument_id, timestamp),
        );

        // Intraday buckets (for pattern recognition)
        let hour_bucket = match dt.hour() {
            0..=6 => 0,   // Overnight
            7..=9 => 1,   // Pre-market/Open
            10..=11 => 2, // Morning
            12..=13 => 3, // Lunch
            14..=15 => 4, // Afternoon
            16..=17 => 5, // Close
            _ => 6,       // After-hours
        };
        features.add("intraday_bucket", hour_bucket as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, TimeZone};

    fn timestamp_from_hms(year: i32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> u64 {
        let dt = NaiveDate::from_ymd_opt(year, month, day)
            .unwrap()
            .and_hms_opt(hour, min, sec)
            .unwrap()
            .and_utc();
        dt.timestamp_micros() as u64
    }

    #[test]
    fn test_session_detection() {
        let config = SessionConfig::default();
        let time_features = TimeFeatures::new(config);

        // Test RTH (2:30 PM UTC = 9:30 AM ET)
        let rth_time = timestamp_from_hms(2024, 3, 15, 14, 30, 0); // Friday
        assert_eq!(
            time_features.get_session(rth_time),
            TradingSession::RegularHours
        );

        // Test pre-market (12:00 PM UTC = 7:00 AM ET)
        let pre_time = timestamp_from_hms(2024, 3, 15, 12, 0, 0);
        assert_eq!(
            time_features.get_session(pre_time),
            TradingSession::PreMarket
        );

        // Test after-hours (9:30 PM UTC = 4:30 PM ET)
        let after_time = timestamp_from_hms(2024, 3, 15, 21, 30, 0);
        assert_eq!(
            time_features.get_session(after_time),
            TradingSession::AfterHours
        );

        // Test weekend
        let weekend_time = timestamp_from_hms(2024, 3, 16, 14, 30, 0); // Saturday
        assert_eq!(
            time_features.get_session(weekend_time),
            TradingSession::Closed
        );
    }

    #[test]
    fn test_time_to_close() {
        let config = SessionConfig::default();
        let time_features = TimeFeatures::new(config);

        // RTH at 3:00 PM UTC (10:00 AM ET), should be 6 hours to close
        let timestamp = timestamp_from_hms(2024, 3, 15, 15, 0, 0);
        let time_to_close = time_features.time_to_session_close(timestamp);
        assert_eq!(time_to_close, Some(6.0 * 3600.0)); // 6 hours in seconds
    }

    #[test]
    fn test_volume_profile() {
        let config = SessionConfig::default();
        let mut time_features = TimeFeatures::new(config);

        let instrument_id = 1;

        // Add some volume data
        for day in 1..=10 {
            for hour in 14..21 {
                // RTH hours
                let timestamp = timestamp_from_hms(2024, 3, day, hour, 0, 0);
                time_features.update_volume(instrument_id, Quantity::from(100u32), timestamp);
            }
            // Force day rollover
            let next_day = timestamp_from_hms(2024, 3, day + 1, 0, 0, 0);
            time_features.update_volume(instrument_id, Quantity::from(1u32), next_day);
        }

        // Check volume percentile
        let test_time = timestamp_from_hms(2024, 3, 15, 17, 0, 0); // 5 PM UTC
        let percentile = time_features.session_volume_percentile(instrument_id, test_time);
        assert!(percentile.is_some());
    }

    #[test]
    fn test_contract_expiry() {
        let config = SessionConfig::default();
        let mut time_features = TimeFeatures::new(config);

        let instrument_id = 1;
        let expiry = Utc.with_ymd_and_hms(2024, 3, 29, 21, 0, 0).unwrap();
        time_features.set_expiry(instrument_id, expiry);

        // Test 14 days before expiry
        let test_time = timestamp_from_hms(2024, 3, 15, 14, 0, 0);
        let days = time_features.days_to_expiry(instrument_id, test_time);
        assert_eq!(days, Some(14));

        // Test roll activity
        let roll_activity = time_features.roll_activity_indicator(instrument_id, test_time);
        assert_eq!(roll_activity, 0.4); // Pre-roll period

        // Test closer to expiry (2 days before)
        let close_time = timestamp_from_hms(2024, 3, 27, 14, 0, 0);
        let close_activity = time_features.roll_activity_indicator(instrument_id, close_time);
        assert_eq!(close_activity, 0.9); // Very close to expiry (<=3 days)
    }

    #[test]
    fn test_feature_vector_integration() {
        let config = SessionConfig::default();
        let mut time_features = TimeFeatures::new(config);

        let instrument_id = 1;
        let timestamp = timestamp_from_hms(2024, 3, 15, 15, 30, 0); // RTH

        // Set up some data
        time_features.set_front_month(instrument_id);
        let expiry = Utc.with_ymd_and_hms(2024, 3, 29, 21, 0, 0).unwrap();
        time_features.set_expiry(instrument_id, expiry);

        let mut features = FeatureVector::new(instrument_id, timestamp);
        time_features.add_to_vector(&mut features, instrument_id, timestamp);

        // Check key features were added
        assert_eq!(features.get("hour_of_day"), Some(15.0));
        assert_eq!(features.get("day_of_week"), Some(4.0)); // Friday
        assert_eq!(features.get("is_rth"), Some(1.0));
        assert_eq!(features.get("is_front_month"), Some(1.0));
        assert!(features.get("seconds_to_session_close").is_some());
        assert!(features.get("days_to_expiry").is_some());
    }
}
