import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.stats as stats_sci
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Analytics Pro", layout="wide")

# --- 2. SIDEBAR INPUTS ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Tickers (3-10):", value="AAPL, MSFT, GOOGL, AMZN")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if len(tickers) > 10:
    st.error("⚠️ Max 10 tickers allowed.")
    st.stop()
elif len(tickers) < 3 and len(tickers) > 0:
    st.warning("⚠️ Minimum 3 tickers required.")
    st.stop()

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_date = col2.date_input("End Date", datetime.now())

if (end_date - start_date).days <= 730:
    st.error("⚠️ Range Error: Minimum 2 years required.")
    st.stop()

rf_rate_annual = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0) / 100

# --- 3. HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    import time
    import yfinance as yf
    
    all_symbols = tickers + ["^GSPC"]
    
    for attempt in range(3):
        try:
            data = yf.download(
                all_symbols,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if not data.empty:
                prices = data['Adj Close']

                # Drop only completely broken tickers
                prices = prices.dropna(axis=1, how='all')

                # Check if any tickers failed
                missing = [t for t in all_symbols if t not in prices.columns]
                if missing:
                    return None, f"Failed to load: {missing}"

                return prices, None

        except Exception as e:
            if attempt == 2:
                return None, f"Data Error: {str(e)}"
            time.sleep(1)

    return None, "Connection timeout. Please try refreshing."

def get_portfolio_stats(weights, returns, rf_annual):
    weights = np.array(weights)
    p_ret = np.sum(returns.mean() * weights) * 252
    p_vol = np.sqrt(weights.T @ (returns.cov() * 252) @ weights)
    sharpe = (p_ret - rf_annual) / p_vol if p_vol != 0 else 0
    downside_diff = returns @ weights - (rf_annual / 252)
    downside_vol = np.sqrt(np.mean(np.minimum(0, downside_diff)**2)) * np.sqrt(252)
    sortino = (p_ret - rf_annual) / downside_vol if downside_vol > 0 else 0
    cum_ret = (1 + (returns @ weights)).cumprod()
    max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    return p_ret, p_vol, sharpe, sortino, max_dd


# --- 4. DATA FETCHING & VALIDATION ---
if not tickers:
    st.info("Please enter 3-10 tickers in the sidebar.")
    st.stop()
else:
    # 1. Fetch data
    df_prices, error = get_data(tickers, start_date, end_date)

    if error:
        st.error(f"❌ Connection Error: {error}")
        st.stop()
        
    elif df_prices is not None:
        # 2. Universal Ticker Check: Remove columns that failed to download
        df_prices = df_prices.dropna(axis=1, how='all')
        
        # 3. Precision Ticker Check
        # Convert columns to a set for a faster, cleaner comparison
        downloaded_tickers = set(df_prices.columns)
        input_tickers = set(tickers)
        
        # This finds ONLY the ones that failed to download
        missing_tickers = list(input_tickers - downloaded_tickers)
        
        if missing_tickers:
            st.error(f"❌ **Incorrect Ticker(s) Detected:** {', '.join(missing_tickers)}")
            st.info("The tickers listed above are not recognized. Please remove or fix them to continue.")
            st.stop()

        # 4. Success: Define clean data for the rest of the app
        df_returns = df_prices.pct_change().dropna()
        stock_list = [t for t in tickers if t in df_prices.columns]
        stock_returns = df_returns[stock_list]
        n_assets = len(stock_list)

        # --- 5. MATH & OPTIMIZATIONS (Required for your tabs) ---
        avg_rets = stock_returns.mean() * 252
        cov_mat = stock_returns.cov() * 252
        init_w = np.array([1. / n_assets] * n_assets)
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # GMV and Tangency Portfolio calculations
        res_gmv = minimize(lambda w: np.sqrt(w.T @ cov_mat @ w), init_w, bounds=bounds, constraints=constraints)
        res_tan = minimize(lambda w: -(np.sum(avg_rets * w) - rf_rate_annual) / np.sqrt(w.T @ cov_mat @ w), init_w, bounds=bounds, constraints=constraints)

        # 6. Sidebar Weight Sliders
        st.sidebar.markdown("---")
        st.sidebar.subheader("Live Portfolio Weights")
        c_raw = []
        for t in stock_list:
            c_raw.append(st.sidebar.slider(f"Weight: {t}", 0.0, 100.0, 100.0/n_assets, key=f"sidebar_slider_{t}"))
        norm_custom = np.array(c_raw) / sum(c_raw) if sum(c_raw) > 0 else init_w

        # 7. Define the Tabs
        tabs = st.tabs(["Exploratory Analysis", "Risk Analysis", "Corr/Cov", "Construction & Optimization", "Portfolio Comparison"])

# --- TAB 1: EXPLORATORY ANALYSIS ---
        with tabs[0]:
            st.header("Exploratory Data Analysis")
            
            # 1. Summary Statistics Table
            st.subheader("Statistical Summary")
            # Calculate descriptive stats and append Skew/Kurtosis
            stats_df = df_returns.describe().T
            stats_df['skewness'] = df_returns.skew()
            stats_df['kurtosis'] = df_returns.kurtosis()
            
            # Reordering columns for better flow: Mean, Std, Min, Max, Skew, Kurt
            stats_display = stats_df[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']]
            st.dataframe(stats_display.style.format("{:.4f}"), use_container_width=True)
            
            st.divider()
            
            # 2. Wealth Index Chart
            st.subheader("Growth of $10,000 (Wealth Index)")
            wealth_index = (1 + df_returns).cumprod() * 10000
            st.line_chart(wealth_index)
            
            st.divider()
            
            # 3. Distribution & Normality Analysis
            st.subheader("Return Distribution & Normality")
            
            dist_col1, dist_col2 = st.columns([1, 3])
            
            with dist_col1:
                # Select Asset
                dist_ticker = st.selectbox("Select Asset to Analyze:", df_returns.columns, key="tab1_asset_sel")
                
                # Select Graph Type (Dropdown added here)
                graph_type = st.selectbox("Select Visualization Type:", 
                                         ["Distribution Plot (Histogram)", "Q-Q Plot (Normality)"], 
                                         key="graph_type_sel")
                
                st.write(f"**Analyzing:** {dist_ticker}")
                if graph_type == "Q-Q Plot (Normality)":
                    st.caption("A Q-Q plot compares the returns to a theoretical normal distribution. If the points fall on the line, the returns are normally distributed.")
                else:
                    st.caption("This shows the frequency of daily returns. Look for 'fat tails' or skewness compared to the bell curve.")

            with dist_col2:
                if graph_type == "Distribution Plot (Histogram)":
                    # Plotly Distplot (Histogram + KDE)
                    fig_dist = ff.create_distplot(
                        [df_returns[dist_ticker].dropna()], 
                        [dist_ticker], 
                        bin_size=.005,
                        show_rug=False
                    )
                    fig_dist.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), height=450)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                else:
                    # Q-Q Plot using Scipy and Plotly
                    # Get the quantiles from scipy
                    (osm, osr), (slope, intercept, r) = stats_sci.probplot(df_returns[dist_ticker].dropna(), dist="norm")
                    
                    fig_qq = go.Figure()
                    
                    # Add the scatter of actual data
                    fig_qq.add_trace(go.Scatter(
                        x=osm, y=osr, 
                        mode='markers', 
                        name='Actual Returns',
                        marker=dict(color='royalblue', size=6, opacity=0.7)
                    ))
                    
                    # Add the reference line (Perfect Normality)
                    line_x = np.array([osm.min(), osm.max()])
                    line_y = slope * line_x + intercept
                    fig_qq.add_trace(go.Scatter(
                        x=line_x, y=line_y, 
                        mode='lines', 
                        name='Normal Distribution',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig_qq.update_layout(
                        title=f"Q-Q Plot: {dist_ticker}",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles",
                        template="plotly_white",
                        height=450,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_qq, use_container_width=True)

# --- TAB 2: RISK ANALYSIS ---
        with tabs[1]:
            st.header("Risk and Volatility Metrics")
            
            # 1. Performance Table
            st.subheader("Annualized Risk-Adjusted Performance")
            risk_metrics = []
            for col in df_returns.columns:
                r, v, sh, so, dd = get_portfolio_stats([1], df_returns[[col]], rf_rate_annual)
                risk_metrics.append({
                    "Asset": col, 
                    "Ann. Return": r, 
                    "Ann. Volatility": v, 
                    "Sharpe Ratio": sh, 
                    "Sortino Ratio": so, 
                    "Max Drawdown": dd
                })
            st.table(pd.DataFrame(risk_metrics).set_index("Asset").style.format("{:.3f}"))
            
            st.divider()
            
            # 2. Drawdown Section (Title Added)
            st.subheader("Drawdown Analysis")
            st.write("This area chart visualizes the peak-to-trough decline in value for the selected asset.")
            dd_asset = st.selectbox("Select Asset for Drawdown View:", df_returns.columns, key="tab2_dd_selectbox")
            
            # Calculate Drawdown series
            wealth_values = (1 + df_returns[dd_asset]).cumprod()
            peak = wealth_values.cummax()
            drawdown_series = (wealth_values - peak) / peak
            
            st.area_chart(drawdown_series)
            st.caption(f"The maximum peak-to-trough decline for {dd_asset} was {drawdown_series.min():.2%}.")
            
            st.divider()
            
            # 3. Rolling Volatility Section (Title Added)
            st.subheader("Rolling Volatility Analysis")
            st.write("Observe how the annualized risk of the assets changes over time.")
            
            vol_col1, vol_col2 = st.columns([1, 3])
            with vol_col1:
                rolling_win = st.slider("Lookback Window (Trading Days):", 20, 252, 60, key="tab2_vol_slider")
                st.info(f"Currently calculating a rolling {rolling_win}-day standard deviation, annualized.")
            
            with vol_col2:
                # Calculate rolling annualized volatility
                rolling_vol = df_returns.rolling(window=rolling_win).std() * np.sqrt(252)
                st.line_chart(rolling_vol)

        # TAB 3: CORRELATION
        with tabs[2]:
            st.subheader("Asset Correlation Matrix")
            st.plotly_chart(px.imshow(df_returns.corr(), text_auto=".2f", color_continuous_scale='RdBu_r'))
            
            st.divider()
            st.subheader("Rolling Correlation Analysis")
            col_a, col_b = st.columns(2)
            asset_a = col_a.selectbox("First Asset", stock_list, index=0, key="corr_a")
            asset_b = col_b.selectbox("Second Asset", stock_list, index=1, key="corr_b")
            if asset_a != asset_b:
                st.line_chart(df_returns[asset_a].rolling(60).corr(df_returns[asset_b]))
            
            with st.expander("Show Daily Covariance Matrix"):
                st.dataframe(df_returns.cov().style.format("{:.6f}"))

# --- TAB 4: CONSTRUCTION & OPTIMIZATION ---
        with tabs[3]:
            st.header("Portfolio Construction & Optimization")
            
            # 1. EQUAL-WEIGHT (1/N) SECTION
            st.subheader("1. Equal-Weight (1/N) Portfolio Performance")
            ew_r, ew_v, ew_sh, ew_so, ew_dd = get_portfolio_stats(init_w, stock_returns, rf_rate_annual)
            
            ew_cols = st.columns(5)
            ew_cols[0].metric("Ann. Return", f"{ew_r:.2%}")
            ew_cols[1].metric("Ann. Volatility", f"{ew_v:.2%}")
            ew_cols[2].metric("Sharpe Ratio", f"{ew_sh:.3f}")
            ew_cols[3].metric("Sortino Ratio", f"{ew_so:.3f}")
            ew_cols[4].metric("Max Drawdown", f"{ew_dd:.2%}")
            
            st.divider()

            # 2. OPTIMIZED PORTFOLIOS (GMV & TANGENCY)
            st.subheader("2. Optimal Portfolios")
            # We explicitly define the stats and charts for both major optimized strategies
            opt_strategies = {
                "Global Minimum Variance (GMV)": res_gmv.x, 
                "Maximum Sharpe Ratio (Tangency)": res_tan.x
            }
            
            for name, w_opt in opt_strategies.items():
                st.markdown(f"#### {name}")
                r_o, v_o, sh_o, so_o, dd_o = get_portfolio_stats(w_opt, stock_returns, rf_rate_annual)
                
                col_metrics, col_chart = st.columns([1, 2])
                with col_metrics:
                    st.table(pd.DataFrame({
                        "Metric": ["Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"], 
                        "Value": [f"{r_o:.2%}", f"{v_o:.2%}", f"{sh_o:.3f}", f"{so_o:.3f}", f"{dd_o:.2%}"]
                    }).set_index("Metric"))
                with col_chart:
                    fig_w = px.bar(
                        pd.DataFrame({"Asset": stock_list, "Weight": w_opt}), 
                        x="Asset", y="Weight", 
                        title=f"Asset Allocation: {name}", 
                        text_auto=".1%", 
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_w.update_layout(yaxis_tickformat=".0%", height=300)
                    st.plotly_chart(fig_w, use_container_width=True)

            st.divider()

            # 3. ESTIMATION WINDOW SENSITIVITY
            st.subheader("3. Estimation Window Sensitivity")
            st.info("**Note:** Historical optimization results are highly sensitive to the input data. Small changes in the lookback period can lead to significantly different portfolio weights.")
            
            t_days = (end_date - start_date).days
            lookback_map = {"Full Sample": t_days}
            if t_days >= 365: lookback_map["Trailing 1 Year"] = 365
            if t_days >= 1095: lookback_map["Trailing 3 Years"] = 1095
            
            selected_lb = st.multiselect(
                "Compare Sensitivity Across Lookbacks:", 
                options=list(lookback_map.keys()), 
                default=list(lookback_map.keys()), 
                key="sens_multi_tab4"
            )
            
            sens_data_list = []
            for label in selected_lb:
                days_lb = lookback_map[label]
                sub_returns = stock_returns.iloc[-days_lb:]
                s_avg = sub_returns.mean() * 252
                s_cov = sub_returns.cov() * 252
                
                # Re-run Tangency optimization for the subset
                s_res_tan = minimize(
                    lambda w: -(np.sum(s_avg * w) - rf_rate_annual) / np.sqrt(w.T @ s_cov @ w), 
                    init_w, bounds=bounds, constraints=constraints
                ).x
                
                sr, sv, ssh, _, _ = get_portfolio_stats(s_res_tan, sub_returns, rf_rate_annual)
                sens_data_list.append({
                    "Lookback Window": label, 
                    "Ann. Return": f"{sr:.2%}", 
                    "Ann. Vol": f"{sv:.2%}", 
                    "Sharpe Ratio": f"{ssh:.3f}"
                })
            
            if sens_data_list:
                st.table(pd.DataFrame(sens_data_list).set_index("Lookback Window"))

            st.divider()

            # 4. EFFICIENT FRONTIER & CAL (With Royal Blue Styling)
            st.subheader("4. Efficient Frontier & CAL")
            st.write("Visualizing the tradeoff between risk and return. The Royal Blue line represents the optimal frontier.")

            # Calculate frontier points
            target_range = np.linspace(avg_rets.min(), avg_rets.max(), 30)
            vols_range = []
            for tr in target_range:
                res = minimize(lambda w: np.sqrt(w.T @ cov_mat @ w), init_w, bounds=bounds, 
                               constraints=[constraints, {'type':'eq','fun':lambda x: np.sum(x*avg_rets)-tr}])
                vols_range.append(res.fun)
            
            fig_ef = go.Figure()

            # The Frontier Line (Royal Blue)
            fig_ef.add_trace(go.Scatter(
                x=vols_range, y=target_range, 
                name="Efficient Frontier", 
                line=dict(color='royalblue', width=4)
            ))

            # Tangency Point (Red Diamond)
            rt_p, vt_p, _, _, _ = get_portfolio_stats(res_tan.x, stock_returns, rf_rate_annual)
            fig_ef.add_trace(go.Scatter(
                x=[vt_p], y=[rt_p], 
                mode='markers+text', 
                text=["Tangency Portfolio"], 
                textposition="top center",
                name="Tangency (Max Sharpe)", 
                marker=dict(size=12, color='red', symbol='diamond')
            ))

            # Custom Portfolio Point (Orange)
            rc_p, vc_p, _, _, _ = get_portfolio_stats(norm_custom, stock_returns, rf_rate_annual)
            fig_ef.add_trace(go.Scatter(
                x=[vc_p], y=[rc_p], 
                mode='markers+text', 
                text=["Custom"], 
                name="Your Custom Portfolio", 
                marker=dict(size=10, color='orange')
            ))

            # Capital Allocation Line (CAL)
            fig_ef.add_trace(go.Scatter(
                x=[0, vt_p*1.5], 
                y=[rf_rate_annual, rf_rate_annual + (rt_p-rf_rate_annual)/vt_p * (vt_p*1.5)], 
                mode='lines', 
                name='CAL', 
                line=dict(dash='dash', color='gray')
            ))

            fig_ef.update_layout(
                xaxis_title="Annualized Volatility (Risk)",
                yaxis_title="Annualized Return",
                template="plotly_white",
                height=550
            )
            st.plotly_chart(fig_ef, use_container_width=True)

            st.divider()

            # 5. RISK CONTRIBUTION (PRC)
            st.subheader("5. Risk Contribution (PRC)")
            prc_sel = st.selectbox("Analyze Portfolio Risk Components:", ["Tangency", "GMV", "Equal-Weight", "Custom"], key="prc_dropdown_tab4")
            w_prc = {"Tangency": res_tan.x, "GMV": res_gmv.x, "Equal-Weight": init_w, "Custom": norm_custom}[prc_sel]
            
            # Marginal Contribution to Risk
            p_variance = w_prc.T @ cov_mat @ w_prc
            marginal_risk = (cov_mat @ w_prc) / np.sqrt(p_variance)
            component_risk = w_prc * marginal_risk
            relative_prc = component_risk / np.sqrt(p_variance)
            
            prc_df = pd.DataFrame({
                "Weight": w_prc, 
                "Risk Contribution": relative_prc
            }, index=stock_list)

            # Grouped Bar Chart for Weight vs. Risk
            fig_prc = px.bar(
                prc_df, 
                barmode='group', 
                labels={"value": "Percentage", "index": "Asset"},
                title=f"Weight vs. Relative Risk Contribution: {prc_sel}",
                color_discrete_sequence=['#636EFA', '#EF553B'] # Blue for Weight, Red for Risk
            )
            fig_prc.update_layout(yaxis_tickformat=".1%", template="plotly_white")
            st.plotly_chart(fig_prc, use_container_width=True)

        # TAB 5: COMPARISON
        with tabs[4]:
            st.header("Portfolio Comparison and Benchmarking")
            all_portfolios = {
                "Equal Weight": init_w, 
                "GMV": res_gmv.x, 
                "Tangency": res_tan.x, 
                "Custom": norm_custom
            }
            
            final_summary = []
            final_wealth = pd.DataFrame(index=stock_returns.index)
            
            for p_name, p_weights in all_portfolios.items():
                pr, pv, psh, pso, pdd = get_portfolio_stats(p_weights, stock_returns, rf_rate_annual)
                final_summary.append({"Portfolio": p_name, "Return": pr, "Volatility": pv, "Sharpe": psh, "Sortino": pso, "Max Drawdown": pdd})
                final_wealth[p_name] = (1 + (stock_returns @ p_weights)).cumprod() * 10000
                
            # Benchmark (S&P 500)
            br, bv, bsh, bso, bdd = get_portfolio_stats([1], df_returns[["^GSPC"]], rf_rate_annual)
            final_summary.append({"Portfolio": "S&P 500", "Return": br, "Volatility": bv, "Sharpe": bsh, "Sortino": bso, "Max Drawdown": bdd})
            final_wealth["S&P 500"] = (1 + df_returns["^GSPC"]).cumprod() * 10000
            
            st.table(pd.DataFrame(final_summary).set_index("Portfolio").style.format("{:.3f}"))
            st.subheader("Comparative Wealth Growth ($10,000 Investment)")
            st.line_chart(final_wealth)