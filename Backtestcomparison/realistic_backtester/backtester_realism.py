import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now use your original imports
from realistic_backtester.Data_loader import DataLoader as Dl
from realistic_backtester.execution_engine import ExecutionEngine as E2
from realistic_backtester.portfolio_manager import Portfolio as Pr
from realistic_backtester.strategy import MACrossover as MA
import matplotlib.pyplot as plt

class OdinsEyeBacktester:
    def __init__(self, initial_capital=10000, region='US_RETAIL'):
        self.portfolio = Pr(initial_capital)
        self.execution = E2(region)
        self.strategy = MA()
        self.data_loader = Dl()
        self.cashb = 0
        self.sharesb = 0
        self.capitalb = initial_capital
        self.portfoliob = []
        self.test = 0
    
    def run_backtest(self, symbol, days_back=500):
        # 1. Load data
    
        data = self.data_loader.load_hourly_data(symbol, days_back)
        
        # 2. Generate signals
        data = self.strategy.generate_signals(data)
        
        # 3. Backtest loop
        for timestamp, row in data.iterrows():
            current_bid = row['bid']
            current_ask = row['ask']
            signal = row['signal']
            mid_price = (current_bid + current_ask) / 2
            
            # Record portfolio value snapshot
            self.portfolio.add_equity_snapshot(
                timestamp=timestamp,
                current_prices={symbol: mid_price}
            )
            
            # BUY SIGNAL
            if signal == 1:
                price = current_ask  # Buy at ASK
                shares = self.portfolio.calculate_position_size(price)
                self.test += 1
                print(self.test)
                print(shares)
                if shares > 0 and self.portfolio.can_buy(symbol, price, shares):
                    execution = self.execution.execute_market_order(
                        symbol=symbol,
                        side='buy',
                        shares=shares,
                        current_bid=current_bid,
                        current_ask=current_ask
                    )
                    
                    
                    if execution:
                        self.portfolio.update(execution)
            
            # SELL SIGNAL (if we hold the stock)
            elif signal == -1 and symbol in self.portfolio.positions:
                if self.portfolio.positions[symbol] > 0:
                    price = current_bid  # Sell at BID
                    shares = self.portfolio.positions[symbol]
                    
                    execution = self.execution.execute_market_order(
                        symbol=symbol,
                        side='sell',
                        shares=shares,
                        current_bid=current_bid,
                        current_ask=current_ask
                    )
                    
                    if execution:
                        self.portfolio.update(execution)
        
        # Return results
        return {
            'equity_curve': self.portfolio.equity_curve,
            'trade_log': self.portfolio.trade_log,
            'final_value': self.portfolio.current_value({symbol: mid_price}),
            'initial_capital': self.portfolio.initial_capital
        }
    def plot_results(self, results, data_with_signals, botdata):
        """
        Plot ODIN'S EYE comparison: Realistic vs Ideal
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[3, 2, 2])
        
        # Plot 1: Price + MA + Signals
        ax1.plot(data_with_signals.index, data_with_signals['mid'], 
                label='Mid Price', alpha=0.7, color='blue', linewidth=1)
        ax1.plot(data_with_signals.index, data_with_signals['ma_fast'], 
                label=f'MA {self.strategy.fast}', alpha=0.7, color='orange')
        ax1.plot(data_with_signals.index, data_with_signals['ma_slow'], 
                label=f'MA {self.strategy.slow}', alpha=0.7, color='red')
        
        # Plot buy/sell signals
        buy_signals = data_with_signals[data_with_signals['signal'] == 1].index
        sell_signals = data_with_signals[data_with_signals['signal'] == -1].index
        
        ax1.scatter(buy_signals, data_with_signals.loc[buy_signals]['mid'], 
                    marker='^', color='green', label='Buy Signal', s=50, alpha=0.7)
        ax1.scatter(sell_signals, data_with_signals.loc[sell_signals]['mid'], 
                    marker='v', color='red', label='Sell Signal', s=50, alpha=0.7)
        
        ax1.set_title('ODIN\'S EYE: Price with MA(50,200) and Signals')
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Equity Curve
        dates = [r['timestamp'] for r in results['equity_curve']]
        values = [r['value'] for r in results['equity_curve']]
        
        ax2.plot(dates, values, label='Realistic Portfolio', color='red', linewidth=2)
        ax2.axhline(y=results['initial_capital'], color='gray', linestyle='--', 
            label=f'Initial Capital (${results["initial_capital"]:,.0f})')
        # Calculate Buy & Hold for comparison
        first_price = data_with_signals['mid'].iloc[0]
        last_price = data_with_signals['mid'].iloc[-1]
        shares_bh = results['initial_capital'] / first_price
        bh_value = shares_bh * last_price
        ax2.plot(dates, botdata, color='green', label=f'Buy & Hold (${bh_value:,.0f})', alpha=0.7)
        
        ax2.set_title('Portfolio Value: Realistic vs Buy & Hold')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        drawdowns = []
        peak = results['initial_capital']
        
        for val in values:
            if val > peak:
                peak = val
            drawdown = (val - peak) / peak * 100  # Percentage drawdown
            drawdowns.append(drawdown)
        
        ax3.fill_between(dates, drawdowns, 0, where=[d < 0 for d in drawdowns], 
                        color='red', alpha=0.3, label='Drawdown')
        ax3.plot(dates, drawdowns, color='darkred', linewidth=1, alpha=0.7)
        ax3.axhline(y=0, color='black', linewidth=0.5)
        
        ax3.set_title('Portfolio Drawdown (%)')
        ax3.set_ylabel('Drawdown %')
        ax3.set_xlabel('Date')
        ax3.legend(loc='lower left')
        ax3.grid(True, alpha=0.3)
        
        # Add stats text
        stats_text = f"""
        Strategy: MA({self.strategy.fast},{self.strategy.slow}) Hourly
        Initial: ${results['initial_capital']:,.0f}
        Final: ${results['final_value']:,.0f}
        Return: {(results['final_value']/results['initial_capital']-1)*100:.2f}%
        Trades: {len(results['trade_log'])}
        Buy & Hold: {((bh_value/results['initial_capital'])-1)*100:.2f}%
        """
        
        fig.text(0.02, 0.02, stats_text, fontsize=9, 
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    def run_and_plot(self, symbol, days_back=500):
        # Run backtest
        results = self.run_backtest(symbol, days_back)
        
        # Get data with signals for plotting
        data = self.data_loader.load_hourly_data(symbol, days_back)
        data_with_signals = self.strategy.generate_signals(data)

        print(f"Initial: ${results['initial_capital']:,.2f}")
        print(f"Final: ${results['final_value']:,.2f}")
        print(f"Trades: {len(results['trade_log'])}")   
        botdata = self.bot_backtest(symbol, days_back=500)
        # Plot
        self.plot_results(results, data_with_signals, botdata)
        
        return results
    def bot_backtest(self, symbol, days_back=500):
        """Simple buy & hold: Buy at first bar, hold through period"""
        data = self.data_loader.load_hourly_data(symbol, days_back)
        
        # Buy at first bar's mid price
        first_mid = (data['bid'].iloc[0] + data['ask'].iloc[0]) / 2
        shares_bh = self.capitalb / first_mid  # All in at start
        
        # Calculate portfolio value at each bar
        portfolio_values = []
        for idx, row in data.iterrows():
            current_mid = (row['bid'] + row['ask']) / 2
            portfolio_value = shares_bh * current_mid
            portfolio_values.append(portfolio_value)
        
        return portfolio_values
    
backtester = OdinsEyeBacktester(initial_capital=1000000)
backtester.run_and_plot('AAPL', days_back=500)



            

        



