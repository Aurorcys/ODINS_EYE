"""
Realistic US market hours using pandas_market_calendars
Provides accurate NYSE schedule for any historical date
"""

import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, time, timedelta, date
import pytz
from pandas.tseries.offsets import CustomBusinessDay



class MarketHours:
    """
    Realistic US market hours using pandas_market_calendars
    Provides accurate NYSE schedule for any historical date
    """
    
    def __init__(self, include_extended_hours=False):
        """
        Args:
            include_extended_hours: If True, consider pre/post market as "open"
                                    Note: Extended hours schedule not in basic mcal
        """
        # Timezones
        self.est = pytz.timezone('US/Eastern')
        self.utc = pytz.UTC
        
        # Get NYSE calendar
        self.nyse = mcal.get_calendar('NYSE')
        
        # Trading hours
        self.regular_open = time(9, 30)
        self.regular_close = time(16, 0)
        self.pre_market_open = time(4, 0)      # Pre-market start (approximate)
        self.post_market_close = time(20, 0)   # Post-market end (approximate)
        
        self.include_extended_hours = include_extended_hours
        
        # Cache for performance
        
        self._schedule_cache = {}
    
    def _get_schedule_for_date(self, date_obj):
        """Get trading schedule for a specific date (cached)"""
        date_str = date_obj.strftime('%Y-%m-%d')
        
        if date_str not in self._schedule_cache:
            try:
                # Get schedule for this single day
                schedule = self.nyse.schedule(start_date=date_str, end_date=date_str)
                self._schedule_cache[date_str] = schedule
            except Exception as e:
                # Fallback: empty schedule
                self._schedule_cache[date_str] = pd.DataFrame()
        
        return self._schedule_cache[date_str]
    
    def _ensure_timezone(self, dt):
        """Ensure datetime has timezone info"""
        if dt.tzinfo is None:
            # Assume it's EST if naive
            dt = self.est.localize(dt)
        return dt.astimezone(self.utc)
    
    def _to_est(self, dt):
        """Convert any datetime to EST"""
        dt = self._ensure_timezone(dt)
        return dt.astimezone(self.est)
    
    def is_market_open(self, dt_input, check_extended=None):
        """
        Check if market is open at given datetime
        
        Args:
            dt_input: datetime (naive or timezone-aware)
            check_extended: Override class default for extended hours
            
        Returns:
            bool: True if market is open
        """
        dt_est = self._to_est(dt_input)
        date_obj = dt_est.date()
        
        # Determine if we're checking extended hours
        use_extended = check_extended if check_extended is not None else self.include_extended_hours
        
        if use_extended:
            # For extended hours, use time-based check
            # Note: mcal doesn't provide extended hours schedule
            return self._is_in_trading_window(dt_est)
        else:
            # For regular hours, check if it's a trading day and within hours
            schedule = self._get_schedule_for_date(date_obj)
    
            if schedule.empty:
                return False  # Market closed all day
    
            # Get market open/close times for this day
            market_open = schedule.iloc[0]['market_open'].tz_convert(self.est)
            market_close = schedule.iloc[0]['market_close'].tz_convert(self.est)
    
            # Check if dt_est is between market open and close
            return market_open <= dt_est <= market_close
    
    def _is_in_trading_window(self, dt_est):
        """
        Fallback method: check if within trading window
        (Used for extended hours or when mcal fails)
        """
        # Get schedule for this date
        schedule = self._get_schedule_for_date(dt_est.date())
    
        if schedule.empty:
            return False  # Market closed all day
    
        current_time = dt_est.time()
    
        # Check if within any trading session
        in_pre = self.pre_market_open <= current_time < self.regular_open
        in_post = self.regular_close <= current_time < self.post_market_close
    
        # For regular hours, just check schedule (NOT 1-min precision)
        if self.regular_open <= current_time < self.regular_close:
            # Use the same logic as is_market_open
            market_open = schedule.iloc[0]['market_open'].tz_convert(self.est)
            market_close = schedule.iloc[0]['market_close'].tz_convert(self.est)
            return market_open <= dt_est <= market_close
    
        # Return True for extended hours if we're including them
        if self.include_extended_hours and (in_pre or in_post):
            return True
    
        return False
    
    def get_market_session(self, dt_input):
        """
        Get current market session
        
        Returns:
            str: 'pre', 'regular', 'post', 'closed', or 'holiday'
        """
        dt_est = self._to_est(dt_input)
        
        # Check if market is completely closed (holiday or weekend)
        schedule = self._get_schedule_for_date(dt_est.date())
        if schedule.empty:
            return 'closed'
        
        current_time = dt_est.time()
        
        # Determine session
        if self.pre_market_open <= current_time < self.regular_open:
            return 'pre'
        elif self.regular_open <= current_time < self.regular_close:
            # On a trading day, between 9:30-4:00 means market is open
            return 'regular'  # Within window but market not open (e.g., lunch break?)
        elif self.regular_close <= current_time < self.post_market_close:
            return 'post'
        else:
            return 'closed'
    
    def get_next_market_open(self, dt_input):
        """
        Get next market opening time
        
        Args:
            dt_input: Starting datetime
            
        Returns:
            datetime: Next market open time (in UTC)
        """
        dt_est = self._to_est(dt_input)
        
        # If currently in market hours, return current time
        if self.is_market_open(dt_input):
            return self._ensure_timezone(dt_est)
        
        # Start searching from tomorrow
        current_date = dt_est.date()
        
        # Search up to 30 days ahead (should always find one)
        for days_ahead in range(1, 31):
            check_date = current_date + timedelta(days=days_ahead)
            schedule = self._get_schedule_for_date(check_date)
            
            if not schedule.empty:
                # Found a trading day, get market open time
                open_time = schedule.iloc[0]['market_open']
                # Convert to EST and set the time
                open_est = open_time.tz_convert(self.est)
                return self._ensure_timezone(open_est)
        
        # Fallback: just add 1 day and set to 9:30 AM
        next_open = dt_est + timedelta(days=1)
        next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
        return self._ensure_timezone(next_open)
    
    def get_close_time(self, dt_input):
        """
        Get the market close time for a specific date
        
        Args:
            dt_input: datetime (for date extraction)
            
        Returns:
            time: Close time for that date
        """
        dt_est = self._to_est(dt_input)
        schedule = self._get_schedule_for_date(dt_est.date())
        
        if schedule.empty:
            return self.regular_close  # Default
        
        # Get close time from schedule
        close_time = schedule.iloc[0]['market_close']
        close_est = close_time.tz_convert(self.est)
        
        return close_est.time()
    
    def get_time_to_close(self, dt_input):
        """
        Get time remaining until market close
        
        Args:
            dt_input: Current datetime
            
        Returns:
            timedelta: Time remaining until close (or 0 if closed)
        """
        dt_est = self._to_est(dt_input)
        
        if not self.is_market_open(dt_input):
            return timedelta(0)
        
        # Get close time for today
        close_time = self.get_close_time(dt_input)
        
        # Create datetime for today's close
        close_dt = datetime.combine(dt_est.date(), close_time)
        close_dt = self.est.localize(close_dt)
        
        # Convert to UTC for comparison
        dt_utc = self._ensure_timezone(dt_est)
        close_utc = close_dt.astimezone(self.utc)
        
        # Return time difference
        time_left = close_utc - dt_utc
        
        # If negative (already past close), return 0
        return max(time_left, timedelta(0))
    
    def convert_hkt_to_est(self, hkt_time):
        """
        Convert Hong Kong time to EST
        
        Args:
            hkt_time: datetime in Hong Kong time (naive or HKT)
            
        Returns:
            datetime: Equivalent time in EST
        """
        hkt_tz = pytz.timezone('Asia/Hong_Kong')
        
        # Localize if naive
        if hkt_time.tzinfo is None:
            hkt_time = hkt_tz.localize(hkt_time)
        
        # Convert to EST
        return hkt_time.astimezone(self.est)
    
    def generate_trading_days(self, start_date, end_date):
        """
        Generate list of trading days between dates
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            list: List of trading day dates
        """
        try:
            # Get schedule for the date range
            schedule = self.nyse.schedule(start_date=start_date, end_date=end_date)
            
            # Extract dates
            trading_days = [idx.date() for idx in schedule.index]
            return trading_days
            
        except Exception as e:
            # Fallback: manual calculation
            print(f"Warning: Using fallback for trading days: {e}")
            all_days = pd.date_range(start=start_date, end=end_date, freq='D')
            
            trading_days = []
            for day in all_days:
                if self.is_market_open(day):
                    trading_days.append(day.date())
            
            return trading_days
    
    def is_early_close(self, dt_input):
        """
        Check if date is an early close day
        
        Args:
            dt_input: datetime or date
            
        Returns:
            bool: True if early close day
        """
        if isinstance(dt_input, datetime):
            dt_est = self._to_est(dt_input)
            date_obj = dt_est.date()
        else:
            date_obj = dt_input
        
        # Check if it's in NYSE early closes
        try:
            # Get early closes for a range around this date
            start_date = date_obj - timedelta(days=1)
            end_date = date_obj + timedelta(days=1)
            early_closes = self.nyse.early_closes(start_date=start_date, end_date=end_date)
            
            return date_obj in [ec.date() for ec in early_closes.index]
        except Exception:
            # Fallback: check if close time is earlier than regular
            close_time = self.get_close_time(date_obj)
            return close_time < self.regular_close

if __name__ == "__main__":
    # First install: pip install pandas_market_calendars
    
    print("Testing Market Hours Module with pandas_market_calendars")
    print("=" * 60)
    
    # Create instance
    mh = MarketHours()
    
    # Test cases
    test_datetimes = [
        # Regular trading
        datetime(2024, 1, 2, 14, 30, 0),   # Tuesday, 2:30 PM EST
        datetime(2024, 1, 2, 4, 30, 0),    # Tuesday, 4:30 AM EST (pre-market)
        
        # Weekend
        datetime(2024, 1, 6, 14, 30, 0),   # Saturday
        
        # Holiday
        datetime(2024, 1, 1, 14, 30, 0),   # New Year's Day
        
        # Early close
        datetime(2024, 11, 29, 13, 30, 0), # Day after Thanksgiving (1:30 PM, market closed)
        
        # Future test
        datetime(2025, 7, 3, 13, 0, 0),    # Day before July 4th, 2025
    ]
    
    for dt in test_datetimes:
        # Localize to EST
        est = pytz.timezone('US/Eastern')
        dt_est = est.localize(dt)
        
        is_open = mh.is_market_open(dt_est)
        next_open = mh.get_next_market_open(dt_est)
        
        print(f"{dt_est.strftime('%Y-%m-%d %H:%M %Z')}:")
        print(f"  Market Open: {'✅' if is_open else '❌'}")
        if not is_open:
            print(f"  Next Open: {next_open.strftime('%Y-%m-%d %H:%M %Z')}")
        print()