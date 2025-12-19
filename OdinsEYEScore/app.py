import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import base64
from datetime import datetime
import plotly.graph_objects as go
from ReturnsAuditor import OdinsEyeAuditor

# Page configuration
st.set_page_config(
    page_title="Odin's Eye - Backtest Auditor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem !important;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚡ Odin\'s Eye</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">See past the illusion of perfect backtests</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Odin%2C_der_G%C3%B6ttervater.jpg/200px-Odin%2C_der_G%C3%B6ttervater.jpg", 
             caption="Odin sacrificed an eye for wisdom")
    
    st.markdown("### 📊 Upload Strategy Data")
    
    upload_method = st.radio(
        "Choose input method:",
        ["CSV File", "Manual Entry", "Sample Data"]
    )
    
    if upload_method == "CSV File":
        uploaded_file = st.file_uploader(
            "Upload CSV with 'returns' column", 
            type=['csv', 'txt'],
            help="File should have a column named 'returns' with decimal returns (e.g., 0.01 for 1%)"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'returns' in df.columns:
                returns = df['returns'].values
                st.success(f"✅ Loaded {len(returns)} returns")
            else:
                st.error("❌ No 'returns' column found")
                returns = None
        else:
            returns = None
            
    elif upload_method == "Manual Entry":
        returns_input = st.text_area(
            "Paste returns (comma or space separated):",
            "0.01, -0.02, 0.015, 0.0, -0.01, 0.02",
            height=100
        )
        if returns_input:
            try:
                returns = np.array([float(x.strip()) for x in returns_input.replace(',', ' ').split()])
                st.success(f"✅ Parsed {len(returns)} returns")
            except:
                st.error("❌ Invalid format")
                returns = None
        else:
            returns = None
            
    else:  # Sample Data
        sample_type = st.selectbox(
            "Choose sample strategy:",
            ["Perfect (Overfit)", "Random Noise", "Realistic", "Trend Following", "Mean Reversion"]
        )
        
        np.random.seed(42)
        if sample_type == "Perfect (Overfit)":
            returns = np.random.normal(0.001, 0.005, 252)
            returns[50:100] += 0.02
        elif sample_type == "Random Noise":
            returns = np.random.normal(0.0, 0.01, 500)
        elif sample_type == "Realistic":
            returns = np.random.normal(0.0003, 0.015, 1000)
        elif sample_type == "Trend Following":
            # Simulate trend following with positive skew
            returns = np.random.normal(0.0005, 0.02, 1000)
            returns = returns + 0.1 * (returns > 0.02) - 0.05 * (returns < -0.02)
        else:  # Mean Reversion
            returns = np.random.normal(0.0002, 0.012, 1000)
            returns = returns * 0.7 + np.roll(returns * -0.3, 1)
        
        st.info(f"Using {sample_type} sample with {len(returns)} periods")
    
    # Benchmark selection
    st.markdown("### 📈 Benchmark Comparison")
    benchmark = st.selectbox(
        "Select benchmark:",
        ["SPY", "QQQ", "IWM", "None"],
        index=0
    )
    
    # Analysis options
    st.markdown("### ⚙️ Analysis Options")
    monte_carlo_sims = st.slider("Monte Carlo Simulations", 100, 10000, 1000, 100)
    confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
    
    st.markdown("---")
    if st.button("🔍 Run Full Audit", type="primary", use_container_width=True):
        st.session_state.run_audit = True
    else:
        st.session_state.run_audit = False

# Main content
if returns is not None and st.session_state.get('run_audit', False):
    # Create auditor
    with st.spinner("Running statistical audit..."):
        auditor = OdinsEyeAuditor(
            returns, 
            benchmark_ticker=benchmark if benchmark != "None" else 'SPY',
            risk_free_rate=0.02
        )
        
        # Get audit results
        audit_results = auditor.audit()
        odin_score = audit_results['odin_score']
        basic_metrics = audit_results['basic_metrics']
        
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = odin_score['score']
        if score > 70:
            color = "#28a745"
            emoji = "✅"
            label = "EXCELLENT"
        elif score > 40:
            color = "#ffc107"
            emoji = "⚠️"
            label = "CAUTION"
        else:
            color = "#dc3545"
            emoji = "🚨"
            label = "DANGER"
        
        st.markdown(f"""
        <div class="score-card" style="background-color: {color}20; border: 2px solid {color};">
            <h1 style="color: {color}; margin: 0; font-size: 3rem;">{score:.0f}</h1>
            <p style="margin: 0; color: {color}; font-size: 1.2rem;">{emoji} {label}</p>
            <p style="margin: 0; font-size: 0.9rem;">Odin's Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Annual Return", f"{basic_metrics['annualized_return']:+.2%}", 
                 delta=f"Sharpe: {basic_metrics['sharpe_ratio']:.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{basic_metrics['max_drawdown']:+.2%}", 
                 delta=f"Return/MaxDD: {basic_metrics['return_over_maxdd']:.2f}")
    
    with col4:
        st.metric("Volatility", f"{basic_metrics['volatility']:.2%}", 
                 delta=f"Cumulative: {basic_metrics['cumulative_return']:+.2%}")
    
    # Generate fingerprint visualization
    st.markdown("## 🎨 Strategy Fingerprint")
    
    with st.spinner("Generating visual fingerprint..."):
        fig = auditor.generate_fingerprint()
        st.pyplot(fig)
        
        # Download button for fingerprint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
            with open(tmp.name, 'rb') as f:
                img_data = f.read()
            
            b64 = base64.b64encode(img_data).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="odin_fingerprint.png">📥 Download Fingerprint</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Component scores
    st.markdown("## 🧪 Component Analysis")
    
    components = odin_score['components']
    cols = st.columns(4)
    
    component_info = {
        'path_dependency': ('🛤️', 'Path Dependency', 'Resistance to lucky sequences'),
        'statistical_significance': ('📊', 'Statistical Significance', 'Signal vs. random noise'),
        'consistency': ('📈', 'Consistency', 'Stability over time'),
        'drawdown_quality': ('📉', 'Drawdown Quality', 'Risk management effectiveness')
    }
    
    for idx, (comp_key, (emoji, name, desc)) in enumerate(component_info.items()):
        with cols[idx]:
            score_val = components[comp_key]
            color = "#28a745" if score_val > 70 else "#ffc107" if score_val > 40 else "#dc3545"
            
            # Create a gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f'{emoji} {name}'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': 'white'},
                    'steps': [
                        {'range': [0, 40], 'color': '#dc3545'},   # Red (was #dc354520)
                        {'range': [40, 70], 'color': '#ffc107'},  # Yellow (was #ffc10720)
                        {'range': [70, 100], 'color': '#28a745'} 
                    ],
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption(desc)
    
    # Detailed findings
    st.markdown("## 🔍 Detailed Audit Findings")
    
    # Path dependency
    path_results = audit_results['path_dependency']
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="color: black;">
            <h4>🛤️ Path Dependency Test</h4>
            <p><strong>Percentile:</strong> {path_results['percentile']:.1%}</p>
            <p><strong>Original Final Value:</strong> {path_results['original_final_value']:.2f}x</p>
            <p><strong>Simulated Median:</strong> {path_results['simulated_median']:.2f}x</p>
            <p><em>{path_results['interpretation']}</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistical significance
    sig_results = audit_results['statistical_significance']
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="color: black;">
            <h4>📊 Statistical Significance</h4>
            <p><strong>p-value vs Zero:</strong> {sig_results['p_value_vs_zero']:.4f}</p>
            <p><strong>Interpretation:</strong></p>
            <p><em>{sig_results['interpretation']}</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Reality checks
    st.markdown("## ⚡ Reality Checks")
    
    reality_checks = []
    
    # Sharpe ratio check
    if basic_metrics['sharpe_ratio'] > 3.0:
        reality_checks.append((
            "danger-box",
            "🚨 Unrealistic Sharpe Ratio",
            f"Sharpe ratio of {basic_metrics['sharpe_ratio']:.2f} is > 3.0. "
            f"99.9% of real strategies have Sharpe < 2.0. This strongly suggests "
            f"data snooping or unrealistic assumptions."
        ))
    elif basic_metrics['sharpe_ratio'] > 2.0:
        reality_checks.append((
            "warning-box",
            "⚠️ Exceptional Sharpe Ratio",
            f"Sharpe ratio of {basic_metrics['sharpe_ratio']:.2f} is exceptional. "
            f"Verify transaction costs, slippage, and ensure no look-ahead bias."
        ))
    
    # Drawdown check
    if abs(basic_metrics['max_drawdown']) < 0.05:  # Less than 5% drawdown
        reality_checks.append((
            "warning-box",
            "⚠️ Suspiciously Low Drawdown",
            f"Max drawdown of {basic_metrics['max_drawdown']:+.2%} is very low for "
            f"the return profile. Real strategies experience larger drawdowns."
        ))
    
    # Return/Vol ratio check
    return_vol_ratio = abs(basic_metrics['annualized_return'] / basic_metrics['volatility'])
    if return_vol_ratio > 1.5:
        reality_checks.append((
            "danger-box",
            "🚨 Impossible Return/Risk Ratio",
            f"Return/Vol ratio of {return_vol_ratio:.2f} > 1.5. Sustainable ratios "
            f"are typically < 1.0. This suggests overfitting or unrealistic assumptions."
        ))
    
    # Display reality checks
    if reality_checks:
        for box_class, title, message in reality_checks:
            st.markdown(f"""
            <div class="{box_class}" style="color: black;">
                <h4>{title}</h4>
                <p>{message}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box" style="color: black;">
            <h4>✅ All Reality Checks Passed</h4>
            <p>Strategy metrics appear within realistic bounds based on historical precedents.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Full text report
    st.markdown("## 📄 Full Text Report")
    with st.expander("View complete audit report", expanded=False):
        report_text = auditor.generate_report()
        st.text(report_text)
        
        # Download button for report
        report_b64 = base64.b64encode(report_text.encode()).decode()
        href = f'<a href="data:text/plain;base64,{report_b64}" download="odin_audit_report.txt">📥 Download Full Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>⚡ Odin's Eye • Backtest Auditor • Version 1.0</p>
        <p><em>"Sacrifice an eye to see the truth"</em></p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Landing page when no analysis has been run
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Odin_riding_Sleipnir.jpg/800px-Odin_riding_Sleipnir.jpg",
                caption="Odin, the Allfather, seeks wisdom beyond illusion")
        
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>See Past the Illusion</h2>
            <p>Upload your strategy's returns to:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>🔍 Detect statistical illusions in backtests</li>
                <li>📊 Calculate the Odin's Score (0-100)</li>
                <li>🎨 Generate a visual strategy fingerprint</li>
                <li>⚡ Run Monte Carlo path analysis</li>
                <li>🚨 Identify overfitting and data snooping</li>
            </ul>
            <p style="margin-top: 2rem;"><strong>Configure your analysis in the sidebar and click "Run Full Audit"</strong></p>
        </div>
        """, unsafe_allow_html=True)