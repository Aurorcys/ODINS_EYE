# test_auditor.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ReturnsAuditor import OdinsEyeAuditor
import numpy as np

def test_perfect_strategy():
    """Test with a 'perfect' strategy (highly unlikely returns)."""
    print("🧪 TEST 1: 'Perfect' Strategy")
    print("-" * 40)
    
    # Create suspiciously perfect returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.005, 252)  # Too smooth
    returns[50:100] += 0.02  # Add a lucky streak
    
    auditor = OdinsEyeAuditor(returns)
    report = auditor.generate_report()
    print(report)
    auditor.plot_equity_curve()

def test_random_strategy():
    """Test with random returns (should score poorly)."""
    print("\n🧪 TEST 2: Random Strategy")
    print("-" * 40)
    
    # Pure random noise
    returns = np.random.normal(0.0, 0.01, 500)
    
    auditor = OdinsEyeAuditor(returns)
    print(auditor.generate_report())

def test_realistic_strategy():
    """Test with realistic strategy returns."""
    print("\n🧪 TEST 3: Realistic Strategy")
    print("-" * 40)
    
    # Create somewhat realistic returns
    np.random.seed(123)
    base_returns = np.random.normal(0.0005, 0.015, 1000)
    # Add some positive drift
    returns = base_returns + 0.0002
    
    auditor = OdinsEyeAuditor(returns, benchmark_ticker='QQQ')
    print(auditor.generate_report())

if __name__ == "__main__":
    test_perfect_strategy()
    test_random_strategy() 
    test_realistic_strategy()