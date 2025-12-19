

import numpy as np


class MACrossover:
    """Simple 20/50 MA crossover"""
    def __init__(self, fast_period=50, slow_period=200):
        self.fast = fast_period
        self.slow = slow_period
    
    def generate_signals(self, data):
        df = data.copy()
        df['mid'] = (df['ask'] + df['bid']) / 2
        
        df['ma_fast'] = df['mid'].rolling(self.fast).mean()
        df['ma_slow'] = df['mid'].rolling(self.slow).mean()
        
        # FIX THE BUG FIRST
        df['signal'] = 0
        df.loc[(df['ma_fast'].shift(1) <= df['ma_slow'].shift(1)) & 
            (df['ma_fast'] > df['ma_slow']), 'signal'] = 1
        df.loc[(df['ma_fast'].shift(1) >= df['ma_slow'].shift(1)) & 
            (df['ma_fast'] < df['ma_slow']), 'signal'] = -1  # NO +1!
        
        # DEBUG
        print(f"🔍 STRATEGY DEBUG:")
        print(f"   MA periods: {self.fast}/{self.slow}")
        print(f"   Buy signals: {(df['signal'] == 1).sum()}")
        print(f"   Sell signals: {(df['signal'] == -1).sum()}")
        print(f"   Total signals: {(df['signal'] != 0).sum()}")
        
        return df