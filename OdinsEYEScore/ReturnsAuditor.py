import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf
from typing import Optional, List, Dict
import matplotlib.pyplot as plt

class OdinsEyeAuditor:
    """Core auditor that analyzes backtest returns for statistical robustness."""
    
    def __init__(self, 
                 strategy_returns: List[float],
                 benchmark_ticker: str = 'SPY',
                 risk_free_rate: float = 0.02):
        """
        Initialize the auditor with strategy returns.
        
        Parameters:
        -----------
        strategy_returns : List[float]
            Periodic returns (daily, weekly, etc.) as decimals (e.g., 0.01 for 1%)
        benchmark_ticker : str
            Ticker for benchmark comparison (default: 'SPY')
        risk_free_rate : float
            Annual risk-free rate for Sharpe calculation
        """
        # Convert to numpy arrays
        self.returns = np.array(strategy_returns)
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate
        
        # Basic validation
        if len(self.returns) < 30:
            print("⚠️ Warning: Less than 30 data points - results may be unreliable")
        
        # Calculate basic metrics
        self._calculate_basic_metrics()
    
    def _calculate_basic_metrics(self):
        """Calculate standard performance metrics."""
        # Cumulative return
        self.cumulative_return = np.prod(1 + self.returns) - 1
        
        # Annualized return (assuming daily returns, 252 trading days)
        self.annualized_return = (1 + self.cumulative_return) ** (252/len(self.returns)) - 1
        
        # Volatility (annualized)
        self.volatility = np.std(self.returns) * np.sqrt(252)
        
        # Sharpe ratio
        self.sharpe_ratio = (self.annualized_return - self.risk_free_rate) / self.volatility
        
        # Max drawdown
        cumulative = (1 + self.returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        self.max_drawdown = np.min(drawdown)
    
    def audit(self) -> Dict:
        """
        Run the full audit and return comprehensive results.
        """
        return {
            'basic_metrics': self._get_basic_metrics(),  # COMMA HERE
            'path_dependency': self._test_path_dependency(),  # COMMA HERE
            'statistical_significance': self._test_statistical_significance(),  # COMMA HERE
            'odin_score': self._calculate_odin_score()  # No comma needed for last item
        }
    
    def _get_basic_metrics(self) -> Dict:
        """Return basic performance metrics."""
        return {
            'cumulative_return': self.cumulative_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'return_over_maxdd': abs(self.annualized_return / self.max_drawdown) if self.max_drawdown != 0 else float('inf')
        }
    
    def _test_path_dependency(self, n_simulations: int = 1000) -> Dict:
        """
        Test if results are dependent on specific sequence of returns.
        
        Uses Monte Carlo reshuffling to see if the strategy's success
        is due to lucky ordering of returns.
        """
        original_final_value = np.prod(1 + self.returns)
        
        # Block bootstrap to preserve some autocorrelation
        block_size = min(5, len(self.returns) // 10)  # Adaptive block size
        n_blocks = len(self.returns) // block_size
        
        simulated_final_values = []
        
        for _ in range(n_simulations):
            # Block bootstrap
            blocks = []
            for i in range(0, len(self.returns) - block_size + 1, block_size):
                blocks.append(self.returns[i:i + block_size])
            
            # Randomly select blocks with replacement
            selected_blocks = np.random.choice(len(blocks), n_blocks, replace=True)
            shuffled_returns = np.concatenate([blocks[i] for i in selected_blocks])
            
            # Trim to original length
            shuffled_returns = shuffled_returns[:len(self.returns)]
            
            # Calculate final value
            simulated_value = np.prod(1 + shuffled_returns)
            simulated_final_values.append(simulated_value)
        
        # Calculate percentile
        simulated_final_values = np.array(simulated_final_values)
        percentile = np.mean(original_final_value > simulated_final_values)
        
        # Score: 100 if exactly median, decreasing as you move away from median
        path_score = 100 * (1 - 2 * abs(percentile - 0.5))
        
        return {
            'score': max(0, path_score),
            'percentile': percentile,
            'interpretation': self._interpret_path_dependency(percentile),
            'original_final_value': original_final_value,
            'simulated_median': np.median(simulated_final_values)
        }
    
    def _interpret_path_dependency(self, percentile: float) -> str:
        """Interpret the path dependency results."""
        if percentile > 0.9:
            return "🚨 EXTREMELY PATH DEPENDENT: Results in top 10% of lucky paths"
        elif percentile > 0.75:
            return "⚠️ Highly path dependent: Results in top 25% of lucky paths"
        elif percentile > 0.6:
            return "⚠️ Somewhat path dependent: Results better than 60% of random paths"
        elif percentile > 0.4:
            return "✅ Reasonably robust: Results near median of random paths"
        else:
            return "✅ Very robust: Results not dependent on specific path"
    
    def _test_statistical_significance(self) -> Dict:
        """
        Test if strategy returns are statistically significant.
        
        Uses t-test to check if mean return is significantly different from zero
        and compares to benchmark if available.
        """
        # Test 1: Is mean return significantly different from zero?
        t_stat_zero, p_value_zero = stats.ttest_1samp(self.returns, 0)
        
        # Test 2: Compare to benchmark if we can fetch it
        benchmark_data = self._fetch_benchmark_data()
        p_value_benchmark = None
        
        if benchmark_data is not None:
            # Align lengths
            min_len = min(len(self.returns), len(benchmark_data))
            returns_aligned = self.returns[:min_len]
            benchmark_aligned = benchmark_data[:min_len]
            
            # Test if strategy beats benchmark
            alpha = returns_aligned - benchmark_aligned
            _, p_value_benchmark = stats.ttest_1samp(alpha, 0)
        
        # Calculate score
        significance_score = 100 * (1 - min(p_value_zero * 10, 1))
        
        return {
            'score': significance_score,
            'p_value_vs_zero': p_value_zero,
            'p_value_vs_benchmark': p_value_benchmark,
            'interpretation': self._interpret_significance(p_value_zero, p_value_benchmark)
        }
    
    def _fetch_benchmark_data(self) -> Optional[np.ndarray]:
        """Fetch benchmark returns using yfinance."""
        try:
            # Download data. The default auto_adjust=True is fine for backtesting[citation:3][citation:4].
            benchmark_df = yf.download(
                self.benchmark_ticker,
                period='1y',
                interval='1d',
                progress=False
            )
            
            if benchmark_df.empty:
                print(f"⚠️ Could not fetch data for {self.benchmark_ticker}")
                return None
                
            
            if isinstance(benchmark_df.columns, pd.MultiIndex):
                # Extract the 'Close' prices for our ticker.
                # This handles the case where data is structured as (Ticker, Metric).
                close_prices = benchmark_df.xs('Close', level=1, axis=1).iloc[:, 0]
            else:
                # If it's a regular DataFrame (shouldn't happen for single ticker with recent yfinance),
                # fall back to standard column access.
                close_prices = benchmark_df['Close']
            
            # Calculate percentage returns and drop the first NaN value
            returns = close_prices.pct_change().dropna().values
            
            # Trim to match the strategy's length if possible, but don't force it
            # Returning full length is safer for alignment in the calling function
            return returns
            
        except Exception as e:
            print(f"⚠️ Could not fetch benchmark data for {self.benchmark_ticker}: {e}")
            return None
    
    def _interpret_significance(self, p_value_zero: float, p_value_benchmark: Optional[float]) -> str:
        """Interpret statistical significance results."""
        if p_value_zero < 0.05:
            result = "✅ Statistically significant vs zero (p < 0.05)"
        else:
            result = "⚠️ Not statistically significant vs zero (could be random)"
        
        if p_value_benchmark is not None:
            if p_value_benchmark < 0.05:
                result += "\n✅ Statistically significant vs benchmark"
            else:
                result += "\n⚠️ Not statistically significant vs benchmark"
        
        return result
    
    def _calculate_odin_score(self) -> Dict:
        """Calculate the overall Odin's Eye Score."""
        # Run all tests
        path_test = self._test_path_dependency()
        sig_test = self._test_statistical_significance()
        
        # Additional simple tests
        consistency_score = self._test_consistency()
        drawdown_score = self._test_drawdown_quality()
        
        # Weighted average
        weights = {
            'path_dependency': 0.35,
            'statistical_significance': 0.35,
            'consistency': 0.15,
            'drawdown_quality': 0.15
        }
        
        total_score = (
            path_test['score'] * weights['path_dependency'] +
            sig_test['score'] * weights['statistical_significance'] +
            consistency_score * weights['consistency'] +
            drawdown_score * weights['drawdown_quality']
        )
        
        # Ensure score is between 0-100
        total_score = max(0, min(100, total_score))
        
        return {
            'score': total_score,
            'components': {
                'path_dependency': path_test['score'],
                'statistical_significance': sig_test['score'],
                'consistency': consistency_score,
                'drawdown_quality': drawdown_score
            },
            'interpretation': self._interpret_overall_score(total_score)
        }
    
    def _test_consistency(self) -> float:
        """Test consistency of returns using rolling Sharpe ratio stability."""
        if len(self.returns) < 60:  # Need enough data for rolling windows
            return 50  # Neutral score
        
        # Calculate rolling Sharpe ratio
        rolling_window = min(60, len(self.returns) // 4)  # 60 periods or 1/4 of data
        returns_series = pd.Series(self.returns)
        
        # Avoid division by zero with a small epsilon
        epsilon = 1e-8
        
        # Rolling mean and std
        rolling_mean = returns_series.rolling(window=rolling_window, min_periods=20).mean()
        rolling_std = returns_series.rolling(window=rolling_window, min_periods=20).std()
        
        # Annualize (assuming daily returns)
        rolling_mean_annualized = rolling_mean * 252
        rolling_std_annualized = rolling_std * np.sqrt(252)
        
        # Rolling Sharpe ratio
        rolling_sharpe = rolling_mean_annualized / (rolling_std_annualized + epsilon)
        
        # Drop NaN values
        rolling_sharpe = rolling_sharpe.dropna()
        
        if len(rolling_sharpe) < 20:
            return 50
        
        # Score based on stability (lower std of rolling Sharpe = more consistent)
        sharpe_std = rolling_sharpe.std()
        
        if sharpe_std == 0:
            return 100  # Perfectly consistent (but suspicious!)
        
        # Convert to score: lower std = higher consistency
        # Typical good strategies have rolling Sharpe std between 0.5-2.0
        if sharpe_std < 0.5:
            consistency_score = 90 + min(10, (0.5 - sharpe_std) * 20)
        elif sharpe_std < 1.0:
            consistency_score = 80 - (sharpe_std - 0.5) * 20
        elif sharpe_std < 2.0:
            consistency_score = 70 - (sharpe_std - 1.0) * 10
        elif sharpe_std < 3.0:
            consistency_score = 50 - (sharpe_std - 2.0) * 10
        else:
            consistency_score = max(10, 40 - (sharpe_std - 3.0) * 5)
        
        return max(10, min(100, consistency_score))
    
    def _test_drawdown_quality(self) -> float:
        """Assess quality of drawdowns (steepness, recovery)."""
        # Calculate Calmar ratio (return / max drawdown)
        if self.max_drawdown != 0:
            calmar_ratio = abs(self.annualized_return / self.max_drawdown)
            # Convert to score: >3 = 100, <0.5 = 0
            drawdown_score = min(100, max(0, (calmar_ratio - 0.5) / 2.5 * 100))
            return drawdown_score
        return 100  # Perfect if no drawdown
    
    def _interpret_overall_score(self, score: float) -> str:
        """Interpret the overall Odin's Score."""
        if score >= 80:
            return "✅ EXCELLENT: Strategy appears robust and statistically sound"
        elif score >= 60:
            return "⚠️ GOOD: Strategy shows promise but has some concerns"
        elif score >= 40:
            return "⚠️ CAUTION: Strategy has significant issues"
        else:
            return "🚨 DANGER: Strategy likely overfit or backtest illusion"
    
    def generate_report(self) -> str:
        """Generate a comprehensive audit report."""
        # Run audit
        audit_results = self.audit()
        odin_score = audit_results['odin_score']
        
        # Create report
        report = []
        report.append("=" * 60)
        report.append("                     ODIN'S EYE AUDIT")
        report.append("=" * 60)
        report.append("")
        
        # Basic metrics
        report.append("📊 BASIC METRICS")
        report.append("-" * 40)
        metrics = audit_results['basic_metrics']
        report.append(f"Cumulative Return:   {metrics['cumulative_return']:+.2%}")
        report.append(f"Annualized Return:   {metrics['annualized_return']:+.2%}")
        report.append(f"Annualized Vol:      {metrics['volatility']:.2%}")
        report.append(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown:        {metrics['max_drawdown']:+.2%}")
        report.append(f"Return/MaxDD:        {metrics['return_over_maxdd']:.2f}")
        report.append("")
        
        # Odin's Score
        score = odin_score['score']
        color = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
        report.append(f"{color} ODIN'S SCORE: {score:.0f}/100")
        report.append(f"Interpretation: {odin_score['interpretation']}")
        report.append("")
        
        # Component scores
        report.append("🧪 COMPONENT SCORES")
        report.append("-" * 40)
        components = odin_score['components']
        for name, comp_score in components.items():
            name_display = name.replace('_', ' ').title()
            bar = "█" * int(comp_score / 10) + "░" * (10 - int(comp_score / 10))
            report.append(f"{name_display:25} {bar} {comp_score:.0f}/100")
        report.append("")
        
        # Detailed findings
        report.append("🔍 DETAILED FINDINGS")
        report.append("-" * 40)
        
        path = audit_results['path_dependency']
        report.append(f"Path Dependency Test:")
        report.append(f"  • Percentile: {path['percentile']:.1%}")
        report.append(f"  • {path['interpretation']}")
        report.append("")
        
        sig = audit_results['statistical_significance']
        report.append(f"Statistical Significance:")
        report.append(f"  • p-value vs zero: {sig['p_value_vs_zero']:.4f}")
        if sig['p_value_vs_benchmark']:
            report.append(f"  • p-value vs benchmark: {sig['p_value_vs_benchmark']:.4f}")
        report.append(f"  • {sig['interpretation']}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_equity_curve(self):
        """Plot the strategy's equity curve."""
        cumulative = (1 + self.returns).cumprod()
        
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative, label='Strategy', linewidth=2)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.3, label='Break-even')
        
        # Highlight drawdowns
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        plt.fill_between(range(len(cumulative)), cumulative, running_max, 
                         where=drawdown < 0, color='red', alpha=0.3, label='Drawdown')
        
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Period')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    
    def generate_fingerprint(self, save_path=None):
        """
        Generate a comprehensive strategy fingerprint visualization.
        Returns matplotlib figure object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from scipy import stats
        import seaborn as sns
        
        # Set professional style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. EQUITY CURVE (Top-left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        cumulative = (1 + self.returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        ax1.plot(cumulative, label='Equity Curve', linewidth=2.5, color='#2E86AB')
        ax1.plot(running_max, '--', label='All-Time High', linewidth=1.5, color='#A23B72', alpha=0.7)
        ax1.fill_between(range(len(cumulative)), cumulative, running_max, 
                        where=drawdown < 0, color='#F18F01', alpha=0.3, label='Drawdown')
        
        # Add annotations for key events
        if len(cumulative) > 20:
            max_point = np.argmax(cumulative)
            min_point = np.argmin(drawdown)
            ax1.scatter(max_point, cumulative[max_point], color='#2E86AB', s=100, zorder=5, 
                    edgecolor='white', linewidth=2, label='Peak')
            ax1.scatter(min_point, cumulative[min_point], color='#F18F01', s=100, zorder=5,
                    edgecolor='white', linewidth=2, label='Max DD Point')
        
        ax1.set_title('Equity Curve & Drawdowns', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Period', fontsize=11)
        ax1.set_ylabel('Cumulative Return', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}x'))
        
        # 2. IMPROVED DRAWDOWN ANALYSIS (FIXED VERSION)
        ax2 = fig.add_subplot(gs[0, 2])

        # Get drawdown data
        cumulative = (1 + self.returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown_series = (cumulative - running_max) / running_max

        # Create the chart
        ax2.fill_between(range(len(drawdown_series)), drawdown_series * 100, 0, 
                        where=drawdown_series < 0, color='#F18F01', alpha=0.6, label='Drawdown')

        # Add key statistical lines (VaR levels)
        percentiles = [0.75, 0.90, 0.95, 0.99]
        colors = ['#28a745', '#ffc107', '#dc3545', '#8b0000']
        labels = ['75% VaR', '90% VaR', '95% VaR', '99% VaR']

        if len(drawdown_series) > 20:
            dd_sorted = np.sort(drawdown_series)
            for pct, color, label in zip(percentiles, colors, labels):
                threshold = dd_sorted[int(pct * len(dd_sorted))]
                ax2.axhline(y=threshold * 100, color=color, linestyle='--', 
                        linewidth=1.5, alpha=0.7, label=f'{label}: {threshold*100:.1f}%')

        # Add max drawdown marker
        max_dd_idx = np.argmin(drawdown_series)
        ax2.scatter(max_dd_idx, drawdown_series[max_dd_idx] * 100, 
                color='red', s=100, zorder=5, edgecolor='white', linewidth=2,
                label=f'Max: {drawdown_series[max_dd_idx]*100:.1f}%')

        # CORRECTED LABELS:
        ax2.set_title('Drawdown Timeline with VaR Levels', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Period', fontsize=11)  # FIXED: Changed from "Percentile" to "Period"
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.legend(fontsize=9, loc='lower left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([drawdown_series.min() * 100 * 1.1, 0])

        # Add stats in corner
        if len(drawdown_series[drawdown_series < 0]) > 0:
            avg_dd = np.mean(drawdown_series[drawdown_series < 0]) * 100
            dd_frequency = len(drawdown_series[drawdown_series < 0]) / len(drawdown_series) * 100
            
            stats_text = f"Avg DD: {avg_dd:.1f}%\nFreq: {dd_frequency:.1f}%"
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. RETURN DISTRIBUTION (Middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Histogram with KDE
        sns.histplot(self.returns * 100, kde=True, ax=ax3, color='#2E86AB', 
                    bins=min(50, len(self.returns)//10), stat='density')
        
        # Overlay normal distribution
        x = np.linspace(self.returns.min() * 100, self.returns.max() * 100, 100)
        normal_pdf = stats.norm.pdf(x, loc=np.mean(self.returns) * 100, 
                                scale=np.std(self.returns) * 100)
        ax3.plot(x, normal_pdf, 'r--', linewidth=2, alpha=0.7, label='Normal Dist')
        
        # Add statistical annotations
        skewness = stats.skew(self.returns)
        kurtosis = stats.kurtosis(self.returns)
        ax3.text(0.05, 0.95, f'Skew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax3.set_title('Return Distribution', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Daily Return (%)', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ROLLING PERFORMANCE (Middle-center)
        ax4 = fig.add_subplot(gs[1, 1])
        
        window = min(60, len(self.returns) // 4)
        rolling_sharpe = pd.Series(self.returns).rolling(window=window).mean() * np.sqrt(252) / \
                        (pd.Series(self.returns).rolling(window=window).std() * np.sqrt(252) + 1e-8)
        rolling_vol = pd.Series(self.returns).rolling(window=window).std() * np.sqrt(252) * 100
        
        # Plot rolling Sharpe
        ax4.plot(rolling_sharpe, label=f'Rolling Sharpe ({window} days)', 
                color='#2E86AB', linewidth=2.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=rolling_sharpe.mean(), color='#A23B72', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Mean: {rolling_sharpe.mean():.2f}')
        
        ax4.set_title(f'Rolling Sharpe Ratio ({window}-day window)', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xlabel('Period', fontsize=11)
        ax4.set_ylabel('Sharpe Ratio', fontsize=11)
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # 5. MONTE CARLO PATHS (Middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Generate Monte Carlo paths
        n_paths = 100
        paths = []
        for _ in range(n_paths):
            # Block bootstrap
            block_size = 5
            blocks = [self.returns[i:i+block_size] for i in range(0, len(self.returns)-block_size+1, block_size)]
            selected = np.random.choice(len(blocks), len(self.returns)//block_size, replace=True)
            shuffled = np.concatenate([blocks[i] for i in selected])[:len(self.returns)]
            paths.append((1 + shuffled).cumprod())
        
        # Plot paths with transparency
        for i, path in enumerate(paths):
            alpha = 0.1 if i < n_paths-1 else 0.8
            color = '#2E86AB' if i < n_paths-1 else '#F18F01'
            linewidth = 0.5 if i < n_paths-1 else 2.5
            label = 'Simulated Paths' if i == 0 else '_nolegend_'
            last_label = 'Original Path' if i == n_paths-1 else '_nolegend_'
            
            ax5.plot(path, color=color, alpha=alpha, linewidth=linewidth, label=label)
        
        ax5.set_title('Monte Carlo Path Analysis', fontsize=14, fontweight='bold', pad=15)
        ax5.set_xlabel('Period', fontsize=11)
        ax5.set_ylabel('Cumulative Return', fontsize=11)
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}x'))
        
        # 6. PERFORMANCE RADAR (Bottom row, spans 3 columns)
        ax6 = fig.add_subplot(gs[2, :])
        
        # Calculate radar metrics
        categories = ['Return', 'Risk Adj', 'Consistency', 'Robustness', 'Drawdown Control']
        
        # Normalize metrics to 0-1 scale
        return_score = min(1.0, self.annualized_return / 0.5)  # Cap at 50% annual return
        sharpe_score = min(1.0, self.sharpe_ratio / 3.0)  # Cap at Sharpe 3.0
        consistency = self._test_consistency() / 100
        robustness = self._test_path_dependency()['score'] / 100
        dd_score = self._test_drawdown_quality() / 100
        
        values = [return_score, sharpe_score, consistency, robustness, dd_score]
        values += values[:1]  # Close the radar
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax6 = plt.subplot(gs[2, :], polar=True)
        ax6.plot(angles, values, 'o-', linewidth=3, color='#2E86AB', markersize=8)
        ax6.fill(angles, values, alpha=0.25, color='#2E86AB')
        
        # Set category labels
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories, fontsize=12, fontweight='bold')
        
        # Set radial labels
        ax6.set_ylim(0, 1)
        ax6.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax6.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax6.grid(True, alpha=1, color='grey')
        
        ax6.set_title('Strategy Performance Radar', fontsize=16, fontweight='bold', pad=30)
        
        # Add overall score annotation
        odin_score = self._calculate_odin_score()['score'] / 100
        ax6.text(0.5, 0.95, f'Odin\'s Score: {odin_score*100:.0f}/100', 
                transform=fig.transFigure, fontsize=18, fontweight='bold',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Color code based on score
        if odin_score > 0.7:
            score_color = '#4CAF50'  # Green
        elif odin_score > 0.4:
            score_color = '#FF9800'  # Orange
        else:
            score_color = '#F44336'  # Red
        
        ax6.text(0.5, 0.90, 
                ['🚨 DANGER', '⚠️ CAUTION', '⚠️ GOOD', '✅ EXCELLENT'][
                    int(odin_score * 4) if odin_score < 1 else 3],
                transform=fig.transFigure, fontsize=14, fontweight='bold',
                horizontalalignment='center', color=score_color)
        
        plt.subplots_adjust(bottom=0.1, top=1.25, hspace=0.80)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
