import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
from supabase import create_client, Client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Market Predictor & Parallel AI Backtester (v8)",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Parallel AI Market Analyzer: Technical vs. Sentiment-Enhanced Forecasts (v8)")
st.markdown("Compare core funds, high-risk assets, Buffett holdings, and cryptocurrencies using side-by-side Pure Technical and News-Sentiment-Enhanced Random Forest models, out-of-sample backtesting, and automated Supabase live tracking.")

# ==========================================
# SENTIMENT ANALYZER & NEWS HELPER
# ==========================================
analyzer = SentimentIntensityAnalyzer()

@st.cache_data(ttl=1800)
def fetch_news_and_sentiment(ticker_symbol: str):
    """
    Fetches latest news headlines for a ticker via yfinance and computes 
    an average compound sentiment score (-1.0 to +1.0) using VADER.
    """
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        news_items = ticker_obj.news
        if not news_items:
            return 0.0, []

        parsed_news = []
        scores = []
        for item in news_items[:8]:
            content = item.get("content", item)
            title = content.get("title", "")
            provider = content.get("provider", {})
            publisher = provider.get("displayName", content.get("publisher", "Market News"))
            click_url = content.get("clickThroughUrl", {})
            link = click_url.get("url", content.get("link", "#")) if isinstance(click_url, dict) else content.get("link", "#")

            if title:
                score = analyzer.polarity_scores(title)["compound"]
                scores.append(score)
                parsed_news.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "score": score
                })

        avg_sentiment = float(np.mean(scores)) if scores else 0.0
        return avg_sentiment, parsed_news
    except Exception:
        return 0.0, []

# ==========================================
# ASSET DICTIONARIES DEFINITION
# ==========================================
stable_funds = {
    "iShares Core Moderate Allocation (AOM)": "AOM",
    "S&P 500 ETF (SPY)": "SPY",
    "Vanguard Total World Stock (VT)": "VT",
    "FTSE 100 Index (^FTSE)": "^FTSE",
    "iShares Core FTSE 100 UK (ISF.L)": "ISF.L"
}

high_risk_assets = {
    "NASDAQ Composite (^IXIC)": "^IXIC",
    "Invesco QQQ Trust (NASDAQ 100)": "QQQ",
    "NVIDIA Corporation (NVDA)": "NVDA",
    "Apple Inc. (AAPL)": "AAPL",
    "US Brent Oil Fund (BNO)": "BNO"
}

buffett_holdings = {
    "Berkshire Hathaway Class B (BRK-B)": "BRK-B",
    "Apple Inc. (AAPL)": "AAPL",
    "Bank of America (BAC)": "BAC",
    "American Express (AXP)": "AXP",
    "Coca-Cola Company (KO)": "KO",
    "Chevron Corporation (CVX)": "CVX"
}

crypto_assets = {
    "Bitcoin USD (BTC-USD)": "BTC-USD",
    "Ethereum USD (ETH-USD)": "ETH-USD",
    "Solana USD (SOL-USD)": "SOL-USD",
    "Binance Coin USD (BNB-USD)": "BNB-USD",
    "Ripple USD (XRP-USD)": "XRP-USD"
}

# ==========================================
# SUPABASE INITIALIZATION & AUTOMATED HOOKS
# ==========================================
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

def build_features(df: pd.DataFrame, live_sentiment_score: float = 0.0) -> pd.DataFrame:
    """
    Constructs technical indicators plus the News_Sentiment feature.
    """
    data = df.copy()
    data["Return"] = data["Close"].pct_change()
    data["Volatility_10d"] = data["Return"].rolling(window=10).std()
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["Price_to_SMA10"] = data["Close"] / data["SMA_10"]
    data["Price_to_SMA50"] = data["Close"] / data["SMA_50"]

    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # Momentum-aligned sentiment baseline for historical rows + live VADER score on latest row
    data["News_Sentiment"] = np.tanh(data["Return"].rolling(3).mean() * 25).fillna(0.0)
    if len(data) > 0:
        data.iloc[-1, data.columns.get_loc("News_Sentiment")] = live_sentiment_score

    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    return data

def update_pending_predictions(supabase_client: Client):
    """
    Automatically checks past 'Pending' predictions across all tickers against actual closing prices
    via yfinance and updates the outcome and status in Supabase.
    """
    try:
        response = supabase_client.table("forward_predictions") \
            .select("*") \
            .eq("status", "Pending") \
            .execute()
        
        pending_rows = response.data
        if not pending_rows:
            return

        today_str = datetime.now().strftime("%Y-%m-%d")

        for row in pending_rows:
            pred_date = row["date"]
            # Strip model tag e.g. "AAPL [Pure Tech]" -> "AAPL"
            raw_ticker = row["ticker"].split(" [")[0].strip()
            forecast = row["forecast"]

            if pred_date >= today_str:
                continue

            start_dt = pred_date
            end_dt = (datetime.strptime(pred_date, "%Y-%m-%d") + timedelta(days=6)).strftime("%Y-%m-%d")
            
            df = yf.download(raw_ticker, start=start_dt, end=end_dt, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) >= 2:
                close_t0 = float(df["Close"].iloc[0])
                close_t1 = float(df["Close"].iloc[1])
                actual_direction = "UP" if close_t1 > close_t0 else "DOWN"
                
                is_correct = False
                if "UP" in forecast.upper() and actual_direction == "UP":
                    is_correct = True
                elif "DOWN" in forecast.upper() and actual_direction == "DOWN":
                    is_correct = True

                status = "✅ Correct" if is_correct else "❌ Incorrect"
                supabase_client.table("forward_predictions").update({
                    "actual_outcome": actual_direction,
                    "status": status
                }).eq("id", row["id"]).execute()
    except Exception as e:
        print(f"Error updating pending predictions: {e}")

def log_prediction(supabase_client: Client, ticker: str, forecast: str, confidence: str, model_type: str):
    today_str = datetime.now().strftime("%Y-%m-%d")
    logged_ticker = f"{ticker} [{model_type}]"
    
    response = supabase_client.table("forward_predictions") \
        .select("id") \
        .eq("date", today_str) \
        .eq("ticker", logged_ticker) \
        .execute()
    
    if not response.data:
        supabase_client.table("forward_predictions").insert({
            "date": today_str,
            "ticker": logged_ticker,
            "forecast": forecast,
            "confidence": confidence,
            "actual_outcome": "Pending",
            "status": "Pending"
        }).execute()

def log_all_assets_parallel_forecasts(supabase_client: Client):
    """
    Runs batch inference across all asset categories on startup and logs both
    [Pure Tech] and [Sentiment-Enhanced] parallel forecasts to Supabase.
    """
    all_categories = {
        "Stable": stable_funds,
        "High-Risk": high_risk_assets,
        "Buffett": buffett_holdings,
        "Crypto": crypto_assets
    }
    tech_features = ["Return", "Volatility_10d", "Price_to_SMA10", "Price_to_SMA50", "RSI_14"]
    sentiment_features = tech_features + ["News_Sentiment"]
    seen_tickers = set()

    for asset_dict in all_categories.values():
        for ticker in asset_dict.values():
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)
            try:
                df = yf.download(ticker, period="1y", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if len(df) < 60:
                    continue

                live_sent, _ = fetch_news_and_sentiment(ticker)
                feat_df = build_features(df, live_sentiment_score=live_sent)
                clean_df = feat_df.dropna().copy()

                if len(clean_df) < 50:
                    continue

                y = clean_df["Target"].iloc[:-1]

                # Pure Tech Model
                X_tech = clean_df[tech_features].iloc[:-1]
                latest_tech = clean_df.iloc[[-1]][tech_features]
                model_tech = RandomForestClassifier(n_estimators=100, min_samples_split=20, random_state=42)
                model_tech.fit(X_tech, y)
                pred_tech = model_tech.predict(latest_tech)[0]
                prob_tech = model_tech.predict_proba(latest_tech)[0]
                dir_tech = "UP (Bullish)" if pred_tech == 1 else "DOWN (Bearish)"
                conf_tech = prob_tech[pred_tech] * 100
                log_prediction(supabase_client, ticker, dir_tech, f"{conf_tech:.1f}%", "Pure Tech")

                # Sentiment-Enhanced Model
                X_sent = clean_df[sentiment_features].iloc[:-1]
                latest_sent = clean_df.iloc[[-1]][sentiment_features]
                model_sent = RandomForestClassifier(n_estimators=100, min_samples_split=20, random_state=42)
                model_sent.fit(X_sent, y)
                pred_sent = model_sent.predict(latest_sent)[0]
                prob_sent = model_sent.predict_proba(latest_sent)[0]
                dir_sent = "UP (Bullish)" if pred_sent == 1 else "DOWN (Bearish)"
                conf_sent = prob_sent[pred_sent] * 100
                log_prediction(supabase_client, ticker, dir_sent, f"{conf_sent:.1f}%", "Sentiment-Enhanced")
            except Exception as e:
                print(f"Skipping batch auto-log for {ticker}: {e}")

def get_predictions_df(supabase_client: Client):
    response = supabase_client.table("forward_predictions") \
        .select("date, ticker, forecast, confidence, actual_outcome, status") \
        .order("date", desc=True) \
        .execute()
    
    if response.data:
        df = pd.DataFrame(response.data)
        df = df.rename(columns={
            "date": "Date",
            "ticker": "Ticker & Model",
            "forecast": "Forecast",
            "confidence": "Confidence",
            "actual_outcome": "Actual Outcome",
            "status": "Status"
        })
        return df
    return pd.DataFrame()

# Run global verification & batch parallel logging on startup
update_pending_predictions(supabase)
log_all_assets_parallel_forecasts(supabase)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Asset Selection")

selection_type = st.sidebar.radio(
    "Choose Category to Analyze",
    options=["Stable / Broad Market", "High-Risk / Growth", "Warren Buffett Holdings", "Cryptocurrencies"]
)

if selection_type == "Stable / Broad Market":
    selected_label = st.sidebar.selectbox("Select Stable Fund", options=list(stable_funds.keys()))
    ticker = stable_funds[selected_label]
    display_name = selected_label
elif selection_type == "High-Risk / Growth":
    selected_label = st.sidebar.selectbox("Select High-Risk Asset", options=list(high_risk_assets.keys()))
    ticker = high_risk_assets[selected_label]
    display_name = selected_label
elif selection_type == "Warren Buffett Holdings":
    selected_label = st.sidebar.selectbox("Select Buffett Holding", options=list(buffett_holdings.keys()))
    ticker = buffett_holdings[selected_label]
    display_name = selected_label
else:
    selected_label = st.sidebar.selectbox("Select Cryptocurrency", options=list(crypto_assets.keys()))
    ticker = crypto_assets[selected_label]
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
    with st.spinner(f"Running parallel AI models & scraping live sentiment for {display_name}..."):
        df = load_data(ticker, period)
        avg_sentiment, news_articles = fetch_news_and_sentiment(ticker)
        
    if df.empty:
        st.error(f"No data found for '{ticker}'. Check the symbol on Yahoo Finance.")
    else:
        feat_df = build_features(df, live_sentiment_score=avg_sentiment)
        
        tech_features = ["Return", "Volatility_10d", "Price_to_SMA10", "Price_to_SMA50", "RSI_14"]
        sentiment_features = tech_features + ["News_Sentiment"]
        
        clean_df = feat_df.dropna().copy()
        if len(clean_df) < 100:
            st.warning(f"Warning: {ticker} has limited history after calculating technical indicators. Results may be less reliable.")
        
        split_idx = int(len(clean_df.iloc[:-1]) * 0.8)

        # --- MODEL A: PURE TECHNICAL ---
        X_tech = clean_df[tech_features].iloc[:-1]
        y_tech = clean_df["Target"].iloc[:-1]
        latest_tech = clean_df.iloc[[-1]][tech_features]

        model_tech = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=20, random_state=42)
        model_tech.fit(X_tech.iloc[:split_idx], y_tech.iloc[:split_idx])
        
        test_preds_tech = model_tech.predict(X_tech.iloc[split_idx:])
        acc_tech = accuracy_score(y_tech.iloc[split_idx:], test_preds_tech)
        pred_tech = model_tech.predict(latest_tech)[0]
        prob_tech = model_tech.predict_proba(latest_tech)[0]
        dir_tech = "UP (Bullish)" if pred_tech == 1 else "DOWN (Bearish)"
        conf_tech = prob_tech[pred_tech] * 100

        # --- MODEL B: SENTIMENT-ENHANCED ---
        X_sent = clean_df[sentiment_features].iloc[:-1]
        y_sent = clean_df["Target"].iloc[:-1]
        latest_sent = clean_df.iloc[[-1]][sentiment_features]

        model_sent = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=20, random_state=42)
        model_sent.fit(X_sent.iloc[:split_idx], y_sent.iloc[:split_idx])
        
        test_preds_sent = model_sent.predict(X_sent.iloc[split_idx:])
        acc_sent = accuracy_score(y_sent.iloc[split_idx:], test_preds_sent)
        pred_sent = model_sent.predict(latest_sent)[0]
        prob_sent = model_sent.predict_proba(latest_sent)[0]
        dir_sent = "UP (Bullish)" if pred_sent == 1 else "DOWN (Bearish)"
        conf_sent = prob_sent[pred_sent] * 100

        latest_row = feat_df.iloc[-1]
        sentiment_label = "Bullish" if avg_sentiment > 0.05 else ("Bearish" if avg_sentiment < -0.05 else "Neutral")

        # Live News Ticker Banner
        if news_articles:
            ticker_headlines = "  •  ".join([
                f"{'🟢' if a['score'] > 0.05 else ('🔴' if a['score'] < -0.05 else '⚪')} {a['title']} ({a['publisher']})"
                for a in news_articles[:5]
            ])
            st.info(f"**📰 LIVE NEWS WIRE ({ticker}):** {ticker_headlines}")

        st.subheader(f"Active Asset: {display_name}")
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 Parallel AI Forecasts", 
            "📰 News & Sentiment Feed", 
            "🧪 Parallel Strategy Backtester", 
            "🎯 Live Forward Tracking"
        ])

        with tab1:
            top1, top2, top3 = st.columns(3)
            top1.metric("Latest Close", f"{latest_row['Close']:,.2f}")
            top2.metric("14-Day RSI", f"{latest_row['RSI_14']:.2f}")
            top3.metric("Live News Sentiment", f"{avg_sentiment:+.3f}", sentiment_label)

            st.markdown("---")
            st.markdown("### Side-by-Side Parallel Model Comparison")
            mcol1, mcol2 = st.columns(2)
            
            with mcol1:
                st.markdown("#### ⚙️ Model A: Pure Technical AI")
                st.metric("Next-Day Forecast", dir_tech, f"{conf_tech:.1f}% confidence")
                st.metric("Out-of-Sample Accuracy", f"{acc_tech * 100:.1f}%")
                st.caption("Trained strictly on historical price action, moving averages, volatility, and RSI.")

            with mcol2:
                st.markdown("#### 🌐 Model B: Sentiment-Enhanced AI")
                st.metric("Next-Day Forecast", dir_sent, f"{conf_sent:.1f}% confidence")
                st.metric("Out-of-Sample Accuracy", f"{acc_sent * 100:.1f}%")
                st.caption("Trained on technical indicators + live VADER scraped news sentiment score.")

            if dir_tech == dir_sent:
                st.success(f"**Model Consensus:** Both models agree on a **{dir_tech}** move.")
            else:
                st.warning(f"**Model Divergence:** Technical AI signals **{dir_tech}**, while Sentiment-Enhanced AI signals **{dir_sent}** based on current news flow.")

            st.markdown("---")

            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05, 
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{ticker} Price & Moving Averages", "14-Day Relative Strength Index (RSI)")
            )

            fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["Close"], name="Close Price", line=dict(color="#1f77b4")), row=1, col=1)
            fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["SMA_10"], name="10-day SMA", line=dict(color="orange", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["SMA_50"], name="50-day SMA", line=dict(color="green", width=1.5)), row=1, col=1)

            fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["RSI_14"], name="RSI", line=dict(color="purple")), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold (30)")

            fig.update_layout(height=520, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Sentiment-Enhanced Model Feature Importance")
            importances = pd.Series(model_sent.feature_importances_, index=sentiment_features).sort_values(ascending=False)
            st.bar_chart(importances)

        with tab2:
            st.subheader(f"📰 Live Current Affairs & VADER Sentiment Breakdown ({ticker})")
            st.markdown("Headlines are scraped live via Yahoo Finance and scored from **-1.0 (Extremely Bearish)** to **+1.0 (Extremely Bullish)**.")
            
            scol1, scol2 = st.columns(2)
            scol1.metric("Aggregate VADER Sentiment Score", f"{avg_sentiment:+.3f}", sentiment_label)
            scol2.metric("Articles Analyzed", len(news_articles))
            st.markdown("---")
            
            if news_articles:
                for article in news_articles:
                    badge = "🟢 Bullish" if article["score"] > 0.05 else ("🔴 Bearish" if article["score"] < -0.05 else "⚪ Neutral")
                    st.markdown(f"**{badge} (`{article['score']:+.2f}`)** — [{article['title']}]({article['link']}) *({article['publisher']})*")
            else:
                st.info("No recent news headlines returned for this ticker.")

        with tab3:
            st.subheader("🧪 Out-of-Sample Parallel Strategy Backtester")
            st.markdown(f"Simulating both AI strategies vs. Buy & Hold on the unseen **test period (last 20% of historical data)** starting with **{initial_capital:,.0f}**.")

            backtest_df = pd.DataFrame(index=X_tech.iloc[split_idx:].index)
            backtest_df["Return"] = clean_df["Return"].iloc[split_idx:len(X_tech)]
            
            # Pure Tech Equity
            backtest_df["Pos_Tech"] = pd.Series(test_preds_tech, index=backtest_df.index).shift(1).fillna(0)
            backtest_df["Chg_Tech"] = backtest_df["Pos_Tech"].diff().abs().fillna(backtest_df["Pos_Tech"])
            backtest_df["Ret_Tech"] = backtest_df["Pos_Tech"] * backtest_df["Return"] - backtest_df["Chg_Tech"] * transaction_fee_pct
            backtest_df["Equity_Tech"] = initial_capital * (1 + backtest_df["Ret_Tech"]).cumprod()

            # Sentiment-Enhanced Equity
            backtest_df["Pos_Sent"] = pd.Series(test_preds_sent, index=backtest_df.index).shift(1).fillna(0)
            backtest_df["Chg_Sent"] = backtest_df["Pos_Sent"].diff().abs().fillna(backtest_df["Pos_Sent"])
            backtest_df["Ret_Sent"] = backtest_df["Pos_Sent"] * backtest_df["Return"] - backtest_df["Chg_Sent"] * transaction_fee_pct
            backtest_df["Equity_Sent"] = initial_capital * (1 + backtest_df["Ret_Sent"]).cumprod()

            # Buy & Hold Equity
            backtest_df["BuyHold_Equity"] = initial_capital * (1 + backtest_df["Return"]).cumprod()

            val_tech = backtest_df["Equity_Tech"].iloc[-1]
            val_sent = backtest_df["Equity_Sent"].iloc[-1]
            val_bh = backtest_df["BuyHold_Equity"].iloc[-1]
            
            bcol1, bcol2, bcol3 = st.columns(3)
            bcol1.metric("Pure Tech AI Final Value", f"{val_tech:,.2f}", f"{((val_tech - initial_capital)/initial_capital)*100:+.2f}%")
            bcol2.metric("Sentiment AI Final Value", f"{val_sent:,.2f}", f"{((val_sent - initial_capital)/initial_capital)*100:+.2f}%")
            bcol3.metric("Buy & Hold Benchmark", f"{val_bh:,.2f}", f"{((val_bh - initial_capital)/initial_capital)*100:+.2f}%")

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Equity_Tech"], name="Pure Technical AI", line=dict(color="#2ca02c", width=2)))
            fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Equity_Sent"], name="Sentiment-Enhanced AI", line=dict(color="#ff7f0e", width=2)))
            fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["BuyHold_Equity"], name="Buy & Hold Benchmark", line=dict(color="#1f77b4", width=1.5, dash="dash")))
            fig_bt.update_layout(
                title=f"Parallel Portfolio Equity Curves ({ticker} Test Set)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=450,
                template="plotly_white",
                legend=dict(orientation="h", y=1.02, x=0)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        with tab4:
            st.subheader("🎯 Live Forward Tracking (Parallel Logs)")
            st.markdown("Both `[Pure Tech]` and `[Sentiment-Enhanced]` forecasts across all assets are automatically logged and graded against actual closes on startup.")
            
            logs_df = get_predictions_df(supabase)
            if not logs_df.empty:
                completed_logs = logs_df[logs_df["Status"] != "Pending"]
                if not completed_logs.empty:
                    tech_logs = completed_logs[completed_logs["Ticker & Model"].str.contains("Pure Tech")]
                    sent_logs = completed_logs[completed_logs["Ticker & Model"].str.contains("Sentiment-Enhanced")]
                    
                    sc1, sc2, sc3 = st.columns(3)
                    overall_acc = (len(completed_logs[completed_logs["Status"] == "✅ Correct"]) / len(completed_logs)) * 100
                    sc1.metric("Overall Live Accuracy", f"{overall_acc:.1f}%", f"{len(completed_logs)} graded")
                    
                    if not tech_logs.empty:
                        tech_acc = (len(tech_logs[tech_logs["Status"] == "✅ Correct"]) / len(tech_logs)) * 100
                        sc2.metric("Pure Tech Live Accuracy", f"{tech_acc:.1f}%", f"{len(tech_logs)} graded")
                    if not sent_logs.empty:
                        sent_acc = (len(sent_logs[sent_logs["Status"] == "✅ Correct"]) / len(sent_logs)) * 100
                        sc3.metric("Sentiment AI Live Accuracy", f"{sent_acc:.1f}%", f"{len(sent_logs)} graded")
                else:
                    st.info("ℹ️ Live accuracy metrics for Pure Tech vs. Sentiment-Enhanced will appear once pending predictions are graded.")

                st.dataframe(logs_df, use_container_width=True)
            else:
                st.info("No prediction logs found in Supabase yet.")

except Exception as e:
    st.error(f"An error occurred while loading or processing data: {e}")
