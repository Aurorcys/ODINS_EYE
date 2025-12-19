import pandas as pd
import numpy as np

class ExecutionEngine:
    def __init__(self, region='US_RETAIL'):
        self.region = region
        self.costs = {
            'US_RETAIL': {'slippage': 0.001, 'commission': 0.005},
            'HK_RETAIL': {'slippage': 0.002, 'commission': 0.008},
            'PRO': {'slippage': 0.0005, 'commission': 0.002}
        }
    def execute_market_order(self, symbol, side, shares, current_bid, current_ask):
        """
        Simple slippage for now: fixed percentage
        """
        params = self.costs[self.region]

        if side == 'buy':
            start_price = current_ask
        else:
            start_price = current_bid
        
        slippage_amount = start_price * params['slippage']
        if side == 'buy':
            execution_price = start_price + slippage_amount
        else:
            execution_price = start_price - slippage_amount
        
        commission = max(shares * params['commission'], 1.0)  # Minimum $1 commission

        fill_prob = .9
        filled = np.random.random() < fill_prob

        if not filled:
            return None  # Order not filled
        
        return {
            'symbol': symbol,
            'execution_price': execution_price,
            'commission': commission,
            'slippage': slippage_amount * shares,
            'filled_shares': shares,
            'side': side,
            'timestamp': pd.Timestamp.now()
        }
