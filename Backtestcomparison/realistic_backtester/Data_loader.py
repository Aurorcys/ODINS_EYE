"""
Realistic Backtester 1.0
"""
from dotenv import load_dotenv
import os
import pandas as pd
import alpaca_trade_api as tradeapi



load_dotenv("ODINSEYE.env")

class DataLoader:
    def __init__(self):
        self.api = tradeapi.REST(
            os.getenv("APCA_API_KEY_ID"),
            os.getenv("APCA_API_SECRET_KEY"),
            'https://paper-api.alpaca.markets'
        )
    
    def load_hourly_data(self, symbol, day_back):
        """
        SIMPLE: Get last N days of 1-hour bars
        No holiday checking - Alpaca already does this
        """
        end_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=500)
        print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
        print(f"📅 Expected bars: ~{day_back * 6.5} trading hours")
        
        # Get bars (Alpaca excludes weekends/holidays automatically)
        bars = self.api.get_bars(
            symbol,
            '1Hour',
            start=start_date.date().isoformat(),
            end=end_date.date().isoformat(),
            adjustment='raw'
        ).df
        
        print(f"📊 Loaded {len(bars)} hourly bars for {symbol}")
        print(f"Date range: {bars.index[0]} to {bars.index[-1]}")
        
        # Add bid/ask (Alpaca doesn't provide these for historical)
        spread_pct = 0.001  # 0.1% realistic spread
        bars['bid'] = bars['close'] * (1 - spread_pct/2)
        bars['ask'] = bars['close'] * (1 + spread_pct/2)
        
        return bars

#bars = api.get_bars('AAPL', '1Hour', start="2022-01-01", end="2024-12-10").df

#print(bars)