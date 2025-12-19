import pandas as pd
import numpy as np

class Portfolio:
    def __init__(self, initial_capital=10000, max_position_pct=0.1):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_pct = max_position_pct
        self.positions = {}  # symbol -> shares
        self.trade_log = []  # List of executed trades
        self.equity_curve = []  # Track portfolio value over time
    def calculate_position_size(self, price, position_pct=0.10):
        """
        Realistic: Buy X% of portfolio
        position_pct: What % of portfolio to allocate (default 10%)
        """
        target_position_value = self.initial_capital * position_pct  # $10,000 * 0.10 = $1,000
        shares = target_position_value / price  # $1,000 / $180 = 5.55 shares
        
        # Real brokers: Can't buy fractional for all stocks
        # AAPL allows fractional, but let's round down
        shares = int(shares)  # 5 shares
        
        # Minimum meaningful position
        min_position = 100  # Don't bother with < $100 trades
        if shares * price < min_position:
            return 0
        
        return max(shares, 0)
    
    def update(self, execution_result):
        """
        Update portfolio after a trade
        """
        symbol = execution_result['symbol']
        shares = execution_result['filled_shares']
        price = execution_result['execution_price']
        commission = execution_result['commission']

        if execution_result['side'] == 'buy':
            self.cash -= (price * shares + commission)
            self.positions[symbol] = self.positions.get(symbol, 0) + shares
        else:
            self.cash += (price * shares - commission)
            self.positions[symbol] = self.positions.get(symbol, 0) - shares

            if abs(self.positions[symbol]) < 0.0001:
                del self.positions[symbol]

        self.trade_log.append(execution_result)
    
    def current_value(self, current_prices):
        """
        Calculate total portfolio value
        """
        total = self.cash
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                total += shares * current_prices[symbol]
        return total
    def add_equity_snapshot(self, timestamp, current_prices):
        """Record portfolio value at a specific time"""
        value = self.current_value(current_prices)
        self.equity_curve.append({
            'timestamp': timestamp,
            'value': value,
            'cash': self.cash,
            'positions': self.positions.copy()
        })
    def can_buy(self, symbol, price, shares):
        """Check if trade is allowed by risk rules"""
        # 2. Enough cash?
        cost = price * shares
        if cost > self.cash * 0.95:  # Keep 5% cash buffer
            return False
        
        # 3. Not too concentrated
        position_value = (self.positions.get(symbol, 0) + shares) * price
        if position_value > self.initial_capital * 0.15:  # Max 15% per stock
            return False
        
        
        return True
    