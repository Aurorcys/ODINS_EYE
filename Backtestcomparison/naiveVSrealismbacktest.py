"""
ODIN'S EYE FINAL COMPARISON
The Truth Revealer: Naive vs Realistic Backtesting
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both backtesters
from Backtestcomparison.naive_backtester import NaiveBacktester
from realistic_backtester.backtester_realism import OdinsEyeBacktester as RealisticBacktester

class OdinsEye:
    """The all-seeing comparison engine"""
    
    def __init__(self, symbol='AAPL', initial_capital=1000000, region='US_RETAIL', days_back=500):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.region = region
        self.days_back = days_back
        
        # Load data once (shared between both backtesters)
        from realistic_backtester.Data_loader import DataLoader
        dl = DataLoader()
        self.data = dl.load_hourly_data(symbol, days_back)
        
        # For naive backtester (uses Close price)
        self.data['Close'] = (self.data['bid'] + self.data['ask']) / 2
            
        print(f"🔍 ODIN'S EYE analyzing {symbol}")
        print(f"   Capital: ${initial_capital:,.0f}")
        print(f"   Period: {days_back} days ({len(self.data)} hourly bars)")
        print(f"   Region: {region}")
        print("=" * 60)
    
    def run_comparison(self):
        """Run both backtests and compare"""
        
        # 1. Run NAIVE backtest (the lie)
        print("\n1. Running NAIVE backtest (The Lie)...")
        naive = NaiveBacktester(initial_capital=self.initial_capital)
        naive_results = naive.backtest(self.data.copy())
        
        print(f"   Naive Return: {naive_results['total_return']:.2f}%")
        print(f"   Naive Trades: {naive_results['trades']}")
        
        # 2. Run REALISTIC backtest (the truth)
        print("\n2. Running REALISTIC backtest (The Truth)...")
        realistic = RealisticBacktester(
            initial_capital=self.initial_capital, 
            region=self.region
        )
        realistic_results = realistic.run_backtest(self.symbol, self.days_back)
        
        realistic_return = (realistic_results['final_value'] / self.initial_capital - 1) * 100
        print(f"   Realistic Return: {realistic_return:.2f}%")
        print(f"   Realistic Trades: {len(realistic_results['trade_log'])}")
        
        # 3. Calculate Buy & Hold
        first_price = self.data['Close'].iloc[0]
        last_price = self.data['Close'].iloc[-1]
        bh_return = (last_price / first_price - 1) * 100
        
        print(f"\n3. Buy & Hold Return: {bh_return:.2f}%")
        
        # 4. Generate comparison visualization
        print("\n4. Generating ODIN'S EYE visualization...")
        self._plot_comparison(naive_results, realistic_results, bh_return)
        
        # 5. Show the brutal truth
        self._print_truth(naive_results, realistic_results, bh_return)
        
        return {
            'naive': naive_results,
            'realistic': realistic_results,
            'buy_hold': bh_return,
            'data': self.data
        }
    
    def _plot_comparison(self, naive_results, realistic_results, bh_return):
        """Create the revealing comparison plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        dates = [r['timestamp'] for r in realistic_results['equity_curve']]
        # Plot 1: Equity Curves
        ax1.plot(dates, naive_results['portfolio_history'], 
                label='Naive (Paper)', color='green', linewidth=2, alpha=0.8)
        
        
        values = [r['value'] for r in realistic_results['equity_curve']]
        ax1.plot(dates, values, 
                label='Realistic', color='red', linewidth=2)
        
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                   label=f'Initial ${self.initial_capital:,.0f}')
        
        ax1.set_title('ODIN\'S EYE: The Great Deception', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Performance Difference (The Cost of Reality)
        min_len = min(len(naive_results['portfolio_history']), len(values))
        naive_aligned = naive_results['portfolio_history'][:min_len]
        realistic_aligned = values[:min_len]
        
        cost_dollars = [n - r for n, r in zip(naive_aligned, realistic_aligned)]
        cost_percent = [((n - r) / n * 100) if n > 0 else 0 
                       for n, r in zip(naive_aligned, realistic_aligned)]
        
        ax2.fill_between(dates[:min_len], cost_dollars, 0, 
                        where=[c > 0 for c in cost_dollars],
                        color='purple', alpha=0.3, label='Execution Cost')
        ax2.plot(dates[:min_len], cost_dollars, 
                color='purple', linewidth=1.5)
        
        ax2.set_title('The Hidden Iceberg: Execution Costs', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cost ($)', color='purple', fontsize=12)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Returns Comparison (Bar Chart)
        returns_data = {
            'Strategy': ['Naive (Paper)', 'Realistic', 'Buy & Hold'],
            'Return %': [
                naive_results['total_return'],
                (realistic_results['final_value'] / self.initial_capital - 1) * 100,
                bh_return
            ],
            'Color': ['green', 'red', 'blue']
        }
        
        bars = ax3.bar(returns_data['Strategy'], returns_data['Return %'], 
                      color=returns_data['Color'], alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('Return Comparison: The Brutal Truth', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Return %', fontsize=12)
        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Trade Statistics
        naive_trades = naive_results['trades']
        realistic_trades = len(realistic_results['trade_log'])
        
        trade_data = {
            'Metric': ['Total Trades', 'Avg Trade Size', 'Win Rate*'],
            'Naive': [naive_trades, f'${self.initial_capital}', 'N/A'],
            'Realistic': [realistic_trades, f'${realistic_results["final_value"]/realistic_trades:,.0f}', 'Calculate from logs']
        }
        
        # Create table
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=pd.DataFrame(trade_data).values,
                         colLabels=pd.DataFrame(trade_data).columns,
                         cellLoc='center', loc='center',
                         colColours=['lightgray']*3)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax4.set_title('Trade Statistics', fontsize=14, fontweight='bold')
        
        # Add summary text
        execution_cost = naive_results['final_value'] - realistic_results['final_value']
        cost_percentage = (execution_cost / naive_results['final_value']) * 100
        
        summary_text = f"""
        THE TRUTH REVEALED:
        
        • Execution costs ate ${execution_cost:,.0f} ({cost_percentage:.1f}%) of paper profits
        • Naive backtest overestimates returns by {(naive_results['total_return'] - ((realistic_results['final_value']/self.initial_capital-1)*100)):.1f}%
        • {realistic_trades} real trades vs {naive_trades} paper trades
        • Real trading turns a {naive_results['total_return']:.1f}% paper profit into a {((realistic_results['final_value']/self.initial_capital-1)*100):.1f}% real return
        
        ODIN'S EYE HAS SPOKEN.
        """
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=10,
                   verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('ODIN\'S EYE: Seeing Through The Paper Trading Lies', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
    def _print_truth(self, naive_results, realistic_results, bh_return):
        """Print the brutal truth"""
        print("\n" + "="*60)
        print("ODIN'S EYE VERDICT:")
        print("="*60)
        
        naive_return = naive_results['total_return']
        realistic_return = (realistic_results['final_value'] / self.initial_capital - 1) * 100
        cost_impact = naive_return - realistic_return
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   Naive (Paper):           {naive_return:>7.2f}%")
        print(f"   Realistic (Actual):      {realistic_return:>7.2f}%")
        print(f"   Buy & Hold:              {bh_return:>7.2f}%")
        
        print(f"\n💰 EXECUTION COST IMPACT:")
        print(f"   Absolute Cost:           ${naive_results['final_value'] - realistic_results['final_value']:,.0f}")
        print(f"   % of Paper Returns:      {cost_impact:.1f}%")
        
        print(f"\n⚡ REALITY CHECK:")
        if realistic_return > bh_return:
            print("   ✅ Strategy BEATS Buy & Hold (after costs!)")
        else:
            print("   ❌ Strategy LOSES to Buy & Hold (costs too high)")
        
        if cost_impact > 5:
            print("   ⚠️  Execution costs >5% - strategy not robust")
        elif cost_impact > 2:
            print("   ⚠️  Execution costs 2-5% - needs careful execution")
        else:
            print("   ✅ Execution costs <2% - strategy is robust")
        
        print(f"\n🎯 ODIN'S WISDOM:")
        print("   'What works on paper often fails in reality.'")
        print("   'The spread, slippage, and commissions are silent killers.'")
        print("   'Test realistically or prepare to lose.'")
        print("="*60)

# Run the comparison
if __name__ == "__main__":
    # Configuration
    SYMBOL = 'AAPL'
    CAPITAL = 1000000  # $1M to see meaningful costs
    REGION = 'US_RETAIL'  # Try 'HK_RETAIL' or 'PRO' for comparison
    DAYS_BACK = 500
    
    print("\n" + "="*60)
    print("WELCOME TO ODIN'S EYE")
    print("The All-Seeing Backtest Comparator")
    print("="*60)
    
    # Create and run comparison
    odin = OdinsEye(
        symbol=SYMBOL,
        initial_capital=CAPITAL,
        region=REGION,
        days_back=DAYS_BACK
    )
    
    results = odin.run_comparison()
    
    print("\n✅ ODIN'S EYE analysis complete!")
    print("   The truth has been revealed.")
    print("\nNext: Try different regions ('HK_RETAIL', 'PRO')")
    print("      or symbols ('SPY', 'TSLA', 'QQQ')")
    print("="*60)