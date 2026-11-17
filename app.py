import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Market Predictor & Backtester",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Multi-Category Market Analyzer & Backtester")
st.markdown("Compare and analyze stable broad-market core funds, high-risk growth assets, and Warren Buffett / Berkshire Hathaway holdings with technical indicators and ML backtesting.")

# Sidebar Controls
st.sidebar.header("Asset Selection")

# Dropdown 1: Stable / Broad Market Funds
stable_funds = {
    "iShares Core Moderate Allocation (AOM)": "AOM",
    "S&P 500 ETF (SPY)": "SPY",
    "Vanguard Total World Stock (VT)": "VT",
    "FTSE 100 Index (^FTSE)": "^FTSE",
    "iShares Core FTSE 100 UK (ISF.L)": "ISF.L"
}

# Dropdown 2: High-Risk / Growth Assets
high_risk_assets = {
    "NASDAQ Composite (^IXIC)": "^IXIC",
    "Invesco QQQ Trust (NASDAQ 100)": "QQQ",
    "NVIDIA Corporation (NVDA)": "NVDA",
    "Apple Inc. (AAPL)": "AAPL",
    "Bitcoin / Crypto Proxy (IBIT)": "IBIT",
    "US Brent Oil Fund (BNO)": "BNO"
}

# Dropdown 3: Warren Buffett / Berkshire Hathaway Holdings & Proxy
buffett_holdings = {
    "Berkshire Hathaway Class B (BRK-B)": "BRK-B",
    "Apple Inc. (AAPL)": "AAPL",
    "Bank of America (BAC)": "BAC",
    "American Express (AXP)": "AXP",
    "Coca-Cola Company (KO)": "KO",
    "Chevron Corporation (CVX)": "CVX"
}

selection_type = st.sidebar.radio(
    "Choose Category to Analyze",
    options=["Stable / Broad Market", "High-Risk / Growth", "Warren Buffett Holdings"]
)

if selection_type == "Stable / Broad Market":
    selected_label = st.sidebar.selectbox("Select Stable Fund", options=list(stable_funds.keys()))
    ticker = stable_funds[selected_label]
    display_name = selected_label
elif selection_type == "High-Risk / Growth":
    selected_label = st.sidebar.selectbox("Select High-Risk Asset", options=list(high_risk_assets.keys()))
    ticker = high_risk_assets[selected_label]
    display_name = selected_label
else:
    selected_label = st.sidebar.selectbox("Select Buffett Holding", options=list(buffett_holdings.keys()))
    ticker = buffett_holdings[selected_label]
    display_name = selected_label

period = st.sidebar.selectbox("Historical Period", options=["1y", "2y", "5y", "max"], index=2)
n_estimators = st.sidebar.slider("Random Forest Estimators", min_value=50, max_value=500, step=50, value=200)

st.sidebar.markdown("---")
st.sidebar.header("Backtest Settings")
initial_capital = st.sidebar.number_input("Starting Capital", value=10000, step=1000, min_value=1000)
transaction_fee_pct = st.sidebar.slider("Transaction Fee per Trade (%)", min_value=0.0, max_value=1.0, step=0.05, value=0.1) / 100

@st.cache_data(ttl=3600)
def load_data(ticker_symbol, hist_period):
    df = yf.download(ticker_symbol, period=hist_period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

try:
    with st.spinner(f"Fetching data and running model for {display_name}..."):
        df = load_data(ticker, period)
        
    if df.empty:
        st.error(f"No data found for '{ticker}'. Check the symbol on Yahoo Finance.")
    else:
        df["Return"] = df["Close"].pct_change()
        df["Volatility_10d"] = df["Return"].rolling(window=10).std()
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["Price_to_SMA10"] = df["Close"] / df["SMA_10"]
        df["Price_to_SMA50"] = df["Close"] / df["SMA_50"]

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        features = ["Return", "Volatility_10d", "Price_to_SMA10", "Price_to_SMA50", "RSI_14"]
        clean_df = df.dropna().copy()
        
        if len(clean_df) < 100:
            st.warning(f"Warning: {ticker} has limited history after calculating technical indicators. Results may be less reliable.")

        X = clean_df[features].iloc[:-1]
        y = clean_df["Target"].iloc[:-1]
        latest_data = clean_df.iloc[[-1]][features]

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=20, random_state=42)
        model.fit(X_train, y_train)

        test_predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, test_predictions)

        next_day_pred = model.predict(latest_data)[0]
        probabilities = model.predict_proba(latest_data)[0]

        latest_row = df.iloc[-1]
        direction = "UP (Bullish)" if next_day_pred == 1 else "DOWN (Bearish)"
        confidence = probabilities[next_day_pred] * 100

        st.subheader(f"Active Asset: {display_name}")
        tab1, tab2 = st.tabs(["📊 Dashboard & ML Forecast", "🧪 Strategy Backtester"])

        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Latest Close", f"{latest_row['Close']:,.2f}")
            col2.metric("14-Day RSI", f"{latest_row['RSI_14']:.2f}")
            col3.metric("Backtest Accuracy", f"{accuracy * 100:.1f}%")
            col4.metric("Next Day Forecast", direction, f"{confidence:.1f}% confidence")

            st.markdown("---")

            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05, 
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{ticker} Price & Moving Averages", "14-Day Relative Strength Index (RSI)")
            )

            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price", line=dict(color="#1f77b4")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_10"], name="10-day SMA", line=dict(color="orange", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="50-day SMA", line=dict(color="green", width=1.5)), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI", line=dict(color="purple")), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold (30)")

            fig.update_layout(
                height=550, 
                margin=dict(l=10, r=10, t=30, b=10), 
                template="plotly_white", 
                legend=dict(orientation="h", y=1.02, x=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Model Feature Importance")
            importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.bar_chart(importances)

        with tab2:
            st.subheader("🧪 Out-of-Sample Strategy Backtest")
            st.markdown(f"Simulating trading performance on the unseen **test period (last 20% of historical data)** starting with **{initial_capital:,.0f}**.")

            backtest_df = X_test.copy()
            backtest_df["Prediction"] = test_predictions
            backtest_df["Return"] = clean_df["Return"].iloc[split_idx:len(X)]
            
            backtest_df["Position"] = backtest_df["Prediction"].shift(1).fillna(0)
            backtest_df["Position_Change"] = backtest_df["Position"].diff().abs().fillna(backtest_df["Position"])

            backtest_df["Strategy_Return"] = (
                backtest_df["Position"] * backtest_df["Return"] 
                - backtest_df["Position_Change"] * transaction_fee_pct
            )

            backtest_df["Strategy_Equity"] = initial_capital * (1 + backtest_df["Strategy_Return"]).cumprod()
            backtest_df["BuyHold_Equity"] = initial_capital * (1 + backtest_df["Return"]).cumprod()

            final_strat_val = backtest_df["Strategy_Equity"].iloc[-1]
            final_bh_val = backtest_df["BuyHold_Equity"].iloc[-1]
            strat_total_return = ((final_strat_val - initial_capital) / initial_capital) * 100
            bh_total_return = ((final_bh_val - initial_capital) / initial_capital) * 100

            bcol1, bcol2, bcol3 = st.columns(3)
            bcol1.metric("Strategy Final Value", f"{final_strat_val:,.2f}", f"{strat_total_return:+.2f}%")
            bcol2.metric("Buy & Hold Final Value", f"{final_bh_val:,.2f}", f"{bh_total_return:+.2f}%")
            bcol3.metric("Outperformance", f"{final_strat_val - final_bh_val:+,.2f}", f"{strat_total_return - bh_total_return:+.2f}%")

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Strategy_Equity"], name="ML Strategy", line=dict(color="#2ca02c", width=2)))
            fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["BuyHold_Equity"], name="Buy & Hold Benchmark", line=dict(color="#1f77b4", width=1.5, dash="dash")))
            
            fig_bt.update_layout(
                title=f"Portfolio Equity Curve ({ticker} Test Set)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=450,
                template="plotly_white",
                legend=dict(orientation="h", y=1.02, x=0)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred while loading or processing data: {e}")
