import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class NaiveBacktester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.shares = 0
        self.trades = 0
        self.portfolio = []
        self.trade_log = []
        self.portfoliob = []
        self.sharesb = 0
        self.capitalb = initial_capital
    def moving_average_crossover(self, data, short_window=20, long_window=50):
        """
        Classic strategy that everyone understands
        Buy when short MA > long MA
        Sell when short MA < long MA
        """
        
        data['short_ma'] = data['Close'].rolling(window=short_window).mean()
        data['long_ma'] = data['Close'].rolling(window=long_window).mean()
        
        
        # Only trade when position changes
        data['signal'] = 0
        data.loc[(data['short_ma'] > data['long_ma']) & (data['short_ma'].shift(1) <= data['long_ma'].shift(1)), 'signal'] = 1
        data.loc[(data['short_ma'] < data['long_ma']) & (data['short_ma'].shift(1) >= data['long_ma'].shift(1)), 'signal'] = -1 
            
        return data
    def execute_trade(self, price, signal, timestamp=None):
        """
        Naive trade execution logic
        """
        if signal == 1 and self.capital > 0:
            shares_to_buy = self.capital / price
            cost = shares_to_buy * price
            self.shares += shares_to_buy
            self.capital -= cost
            self.trades += 1

            self.trade_log.append({
                'timestamp': timestamp,
                'action': 'BUY',
                'price': price,
                'shares': shares_to_buy,
                'cost': cost,
                'note': 'Naive: perfect fill at mid-price'
            })

        elif signal == -1 and self.shares > 0:
            proceeds = self.shares * price
            self.capital += proceeds
            shares_sold = self.shares
            self.shares = 0
            self.trades += 1

            self.trade_log.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': price,
                'shares': shares_sold,
                'proceeds': proceeds,
                'note': 'Naive: perfect fill at mid-price'
            })
        





    def backtest(self, data):
        """
        Run the naive backtest
        """
        self.capital = self.initial_capital
        self.shares = 0
        self.trades = 0
        self.portfolio = []
        self.trade_log = []
        data = self.moving_average_crossover(data)

        for i in range(len(data)):
            if i < 50:
                portfolio_value = self.initial_capital
            else:  # Skip initial period for MA calculation

                price = float(data['Close'].iloc[i].item())
                signal = float(data['signal'].iloc[i])

                if not np.isnan(signal) and signal != 0:
                    self.execute_trade(price, signal, data.index[i])
                
                portfolio_value = self.capital + (self.shares * price)
            self.portfolio.append(portfolio_value)

        return {
            'initial_capital': self.initial_capital,
            'final_value': self.portfolio[-1] if self.portfolio else self.initial_capital,
            'total_return': ((self.portfolio[-1] - self.initial_capital) / self.initial_capital * 100) if self.portfolio else 0,
            'trades': self.trades,
            'trade_log': pd.DataFrame(self.trade_log),
            'portfolio_history': self.portfolio
        }
    
    def buy_hold(self, data):
        """
        Buy and Hold Strategy for comparision
        """
        self.sharesb = 0
        self.capitalb = self.initial_capital
        self.portfoliob = []
        for i in range(len(data)):
            price = float(data['Close'].iloc[i].item())
            if i == 0:
                shares_to_buy = self.capitalb / price
                self.sharesb += shares_to_buy
                self.capitalb -= shares_to_buy * price
            portfolio_valueb = self.capitalb + (self.sharesb * price)
            self.portfoliob.append(portfolio_valueb)
        data = data.copy()
        data['buy_hold_portfolio'] = self.portfoliob
        return data

    def plot_results(self, data):
        """
        PLOTTING RESULTS
        """

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 2])

        ax1.plot(data['Close'], label='Close Price', alpha=0.7, color='blue')
        ax1.plot(data['short_ma'].iloc[50:], label=f'MA {20}', alpha=0.7, color='orange')
        ax1.plot(data['long_ma'].iloc[50:], label=f'MA {50}', alpha=0.7, color='red')

        buy_signals = data[data['signal'] == 1].index
        sell_signals = data[data['signal'] == -1].index

        ax1.scatter(buy_signals, data.loc[buy_signals]['Close'], marker='^', color='green', label='Buy Signal', s=100)
        ax1.scatter(sell_signals, data.loc[sell_signals]['Close'], marker='v', color='red', label='Sell Signal', s=100)

        ax1.set_title('Price with Moving Averages and Trade Signals (NAIVE BACKTESTER)')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(data.index, self.portfolio, label='Portfolio Value', color='red')
        ax2.plot(data.index, data['buy_hold_portfolio'], label='Buy & Hold Portfolio', color='gray')
        ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_ylabel('Portfolio Value')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print('Running Naive Backtester')
    data = yf.download(input('Please select a ticker: ').upper(), period='2y', progress=False)

    backtester = NaiveBacktester(initial_capital=10000)
    results = backtester.backtest(data)
    data = backtester.buy_hold(data)
    print(f'\n{'='*60}')
    print('Backtest Results: (NAIVE BACKTESTER: THE LIE)')
    print(f'\n{'='*60}')
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Portfolio Value: ${results['final_value']:.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Total Trades Executed: {results['trades']}")
    print(f'\n{'='*60}\n')
    print(f'Initial Capital for B&H: ${backtester.initial_capital:.2f}')
    print(f'Final Portfolio Value for B&H: ${backtester.portfoliob[-1]:.2f}')
    total_return_bh = ((backtester.portfoliob[-1] - backtester.initial_capital) / backtester.initial_capital * 100)
    print(f'Total Return for B&H: {total_return_bh:.2f}%')
    print(f'\n{'='*60}\n')
    if not results['trade_log'].empty:
        print(f"\nTrade Log:")
        print(results['trade_log'].to_string(index=False))
    print(f'\n{'='*60}')
    print('WARNING: This backtester assumes perfect trade execution at mid-price without slippage or fees. Assuming false perfect conditions.')
    
    backtester.plot_results(data)

