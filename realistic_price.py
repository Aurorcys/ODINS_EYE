"""
REALISTIC PRICE SIMULATION
This shows why naive backtesters lie about profits.
"""

from alpaca.data import StockHistoricalDataClient, StockLatestQuoteRequest
from alpaca.data import StockTradesRequest, TimeFrame
from alpaca.trading import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os
from datetime import datetime, timedelta, time
from typing import Tuple, Optional

API_KEY = 'PKQU2YWIR63CWRUNV7WFTLMSJT'
SECRET_KEY = 'AgEn1o6Xqh7ZdUwjqpHbUacuRPvMWa5H1PRqa21EQJRy'

class AlpacaRealData:
    """
    Gets REAL bid/ask from Alpaca Markets.
    Paper trading = FREE real-time data.
    """
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.has_credentials = bool(api_key and secret_key)
        
        if self.has_credentials:
            # Data client for market data
            self.data_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key
            )
            # Trading client for orders (future use)
            self.trading_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=True  # Paper trading!
            )
            print("✅ Connected to Alpaca Paper Trading")
        else:
            print("⚠️  No Alpaca API keys. Using simulated data.")
            self.data_client = None
    
    def get_real_bid_ask(self, symbol: str) -> Optional[Tuple[float, float, float]]:
        """
        Get ACTUAL bid and ask prices from Alpaca.
        Returns: (bid_price, ask_price, spread) or None
        """
        if not self.has_credentials:
            print("No API keys. Getting simulated spread instead...")
            return self._get_simulated_spread(symbol)
        
        try:
            # Request latest quote
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes and quotes[symbol]:
                quote = quotes[symbol]
                bid = quote.bid_price
                ask = quote.ask_price
                
                if bid and ask:
                    spread = ask - bid
                    return bid, ask, spread
                else:
                    print(f"No bid/ask data for {symbol}")
                    return None
            else:
                print(f"No quote data for {symbol}")
                return None
                
        except Exception as e:
            print(f"Alpaca API error: {e}")
            return None
    
    def _get_simulated_spread(self, symbol: str) -> Tuple[float, float, float]:
        """
        Simulate realistic spread when no API access.
        Based on actual market behavior.
        """
        import yfinance as yf
        
        # Get current price
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        
        if hist.empty:
            # Default fallback
            bid, ask = 100.0, 100.1
        else:
            current_price = hist['Close'].iloc[-1]
            
            # Realistic spread simulation:
            # Liquid stocks (AAPL, SPY): 0.05-0.1%
            # Less liquid: 0.2-0.5%
            # Based on actual market observations
            if symbol in ["AAPL", "SPY", "QQQ", "MSFT"]:
                spread_pct = 0.001  # 0.1%
            else:
                spread_pct = 0.002  # 0.2%
            
            spread_amount = current_price * spread_pct
            bid = current_price - (spread_amount / 2)
            ask = current_price + (spread_amount / 2)
        
        spread = ask - bid
        return bid, ask, spread
    def is_nyse_open_from_hkt(self) -> bool:
    #convert HKT to EST
        now_hkt = datetime.now()
        est_time = now_hkt - timedelta(hours=13)  # HKT → EST
    
    # Check if weekday
        if est_time.weekday() >= 5:  # Weekend
            return False
    
    # Check if between 9:30 AM and 4:00 PM EST
        est_time_only = est_time.time()
        return time(9, 30) <= est_time_only <= time(16, 0)
    
    def get_spread_warning(self, spread_pct: float, is_market_open: bool) -> str:
        if self.is_nyse_open_from_hkt():
        # Normal market hours
            if spread_pct < 0.1:  # 0.1%
                return "🟢 Normal spread"
            elif spread_pct < 0.5:  # 0.5%
                return "🟡 Elevated spread - consider waiting"
            else:
                return "🔴 Excessive spread - DO NOT TRADE"
        else:
        # After/pre-market hours
            if spread_pct < 1.0:  # 1.0%
                return "🟡 Normal after-hours spread"
            elif spread_pct < 5.0:  # 5.0%
                return "🔴 High after-hours spread - risky"
            else:
                return "🚨 EXTREME spread - likely bad data or illiquid"
    
    def calculate_trading_cost(self, symbol: str, shares: int = 100) -> None:
        """
        Show REAL trading costs with Alpaca data.
        """
        print(f"\n{'='*60}")
        print(f"ALPACA REAL TRADING COST ANALYSIS: {symbol}")
        print(f"{'='*60}")
        
        result = self.get_real_bid_ask(symbol)
        
        if not result:
            print("❌ Failed to get market data.")
            return
            
        bid, ask, spread = result
        
        if self.is_nyse_open_from_hkt():
            print("🟢 NYSE is OPEN now.")
        else:
            print("🔴 NYSE is CLOSED now.(THERE IS A HIGHER SPREAD WHEN PREMARKET)")
        
        spread_warning = self.get_spread_warning((spread/bid)*100, self.is_nyse_open_from_hkt())
        print(f"⚠️  SPREAD ANALYSIS: {spread_warning}")

        print(f"\n📈 MARKET DATA:")
        print(f"   Symbol:          {symbol}")
        print(f"   Bid (Sell at):   ${bid:.2f}")
        print(f"   Ask (Buy at):    ${ask:.2f}")
        print(f"   Spread:          ${spread:.4f}")
        print(f"   Spread (%):      {(spread/bid*100):.4f}%")
        
        # Trading cost analysis
        print(f"\n💰 TRADING COST (for {shares:,} shares):")
        
        # Round trip cost
        buy_cost = ask * shares
        sell_proceeds = bid * shares
        spread_cost = (ask - bid) * shares
        
        print(f"   Buy {shares} @ Ask:    ${buy_cost:,.2f}")
        print(f"   Sell {shares} @ Bid:   ${sell_proceeds:,.2f}")
        print(f"   Spread Cost:          ${spread_cost:,.2f}")
        print(f"   Effective Cost/Share: ${spread_cost/shares:.4f}")
        
        # Commission (Alpaca = $0 for stocks)
        commission = 0
        print(f"   Commission:           ${commission:.2f} (Alpaca = free)")
        
        # Total cost
        total_cost = spread_cost + commission
        print(f"\n   TOTAL COST:           ${total_cost:,.2f}")
        
        # Naive backtest comparison
        mid_price = (bid + ask) / 2
        naive_profit = 0  # Buy and sell at same price
        real_profit = -total_cost  # Negative = loss
        
        print(f"\n📊 BACKTEST REALITY CHECK:")
        print(f"   Naive backtest profit:  ${naive_profit:,.2f}")
        print(f"   Realistic profit:       ${real_profit:,.2f}")
        print(f"   Difference:             ${abs(real_profit):,.2f} LOSS")
        
        if self.has_credentials:
            print(f"\n✅ Using REAL Alpaca market data")
        else:
            print(f"\n⚠️  Using SIMULATED spread (get API keys for real data)")
    
    def show_how_to_get_keys(self):
        """Instructions to get Alpaca API keys."""
        print("\n" + "="*60)
        print("HOW TO GET ALPACA API KEYS (FREE):")
        print("="*60)
        print("1. Go to: https://app.alpaca.markets/signup")
        print("2. Sign up for PAPER TRADING account (100% free)")
        print("3. Go to Dashboard → API Keys")
        print("4. Generate new keys or use existing ones")
        print("5. Copy:")
        print("   - API Key ID (starts with PK...)")
        print("   - Secret Key")
        print("\n⚠️  NEVER share these keys or commit to GitHub!")
        print("   Use environment variables instead.")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Initialize with your keys (or empty for simulation)
    trader = AlpacaRealData(api_key=API_KEY, secret_key=SECRET_KEY)
    
    # Analyze a stock
    symbol = "AAPL"  # Change to any: "TSLA", "NVDA", "SPY"
    
    trader.calculate_trading_cost(symbol, shares=100)
    
    # Show how to get real data
    if not trader.has_credentials:
        trader.show_how_to_get_keys()
    
    print(f"\n{'='*60}")
    print("This spread cost is what God's Eye will account for.")
    print("Most backtesters ignore it → show fake profits.")
    print("Your backtester will show REAL profits/losses.")
    print(f"{'='*60}")