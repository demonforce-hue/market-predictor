import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from supabase import create_client

# --- PAGE CONFIG ---
st.set_page_config(page_title="Market AI Multi-Model Companion v10", page_icon="📈", layout="wide")

# --- SUPABASE SETUP ---
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# --- ASSET CATEGORIES & PORTFOLIOS (WITH DESCRIPTIVE NAMES) ---
ASSET_DICTIONARY = {
    "🪙 Crypto / Digital Assets": {
        "BTC-USD": "Bitcoin USD",
        "ETH-USD": "Ethereum USD",
        "SOL-USD": "Solana USD",
        "AVAX-USD": "Avalanche USD",
        "COIN": "Coinbase Global Inc."
    },
    "🥇 Commodities & Safe Havens": {
        "GLD": "SPDR Gold Shares",
        "SLV": "iShares Silver Trust",
        "USO": "United States Oil Fund",
        "UNG": "United States Natural Gas Fund",
        "DBA": "Invesco DB Agriculture Fund"
    },
    "🔥 High-Risk / Growth": {
        "TQQQ": "ProShares UltraPro QQQ",
        "SOXL": "Direxion Daily Semiconductor Bull 3X",
        "ARKK": "ARK Innovation ETF",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla Inc."
    },
    "🛡️ Stable Broad Market": {
        "SPY": "SPDR S&P 500 ETF Trust",
        "QQQ": "Invesco QQQ Trust",
        "VTI": "Vanguard Total Stock Market ETF",
        "IVV": "iShares Core S&P 500 ETF",
        "DIA": "SPDR Dow Jones Industrial Average ETF"
    },
    "📊 Top 5 Performing ETFs": {
        "SPY": "SPDR S&P 500 ETF Trust",
        "QQQ": "Invesco QQQ Trust",
        "IWM": "iShares Russell 2000 ETF",
        "XLF": "Financial Select Sector SPDR Fund",
        "XLE": "Energy Select Sector SPDR Fund"
    },
    "💼 iShares Moderate / Core": {
        "AOM": "iShares Core Moderate Allocation ETF",
        "AOR": "iShares Core Growth Allocation ETF",
        "AGG": "iShares Core U.S. Aggregate Bond ETF",
        "IEFA": "iShares Core MSCI EAFE ETF",
        "IEMG": "iShares Core MSCI Emerging Markets ETF"
    },
    "🌍 Global Indexes": {
        "^DJI": "Dow Jones Industrial Average",
        "^FTSE": "FTSE 100 Index",
        "^N225": "Nikkei 225 Index",
        "^GSPC": "S&P 500 Index"
    }
}

# --- HELPER FUNCTIONS ---
def fetch_stock_data(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def add_technical_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_news_sentiment(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news
        if not news:
            return 0.0
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for item in news[:5]:
            title = item.get('title', '')
            score = analyzer.polarity_scores(title)['compound']
            scores.append(score)
        return np.mean(scores) if scores else 0.0
    except Exception:
        return 0.0

# --- MODEL TRAINING & PREDICTION ---
def run_models(ticker):
    df = fetch_stock_data(ticker)
    if len(df) < 60:
        return None, None, None, "Insufficient data (need at least 60 days of history)."
    
    df = add_technical_indicators(df)
    feature_cols = ['SMA_10', 'SMA_50', 'RSI', 'Volatility']
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    model_df = df.dropna(subset=feature_cols + ['Target'])
    
    X = model_df[feature_cols]
    y = model_df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    latest_feat = X.tail(1)
    
    # 1. Pure Technical Random Forest
    rf_pure = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_pure.fit(X_train, y_train)
    rf_pure_pred = rf_pure.predict(latest_feat)[0]
    rf_pure_proba = max(rf_pure.predict_proba(latest_feat)[0]) * 100
    pure_dir = "UP" if rf_pure_pred == 1 else "DOWN"
    
    # 2. Sentiment-Enhanced Random Forest
    sentiment_score = get_news_sentiment(ticker)
    X_train_sent = X_train.copy()
    latest_feat_sent = latest_feat.copy()
    X_train_sent['Sentiment'] = sentiment_score
    latest_feat_sent['Sentiment'] = sentiment_score
    
    rf_sent = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_sent.fit(X_train_sent, y_train)
    rf_sent_pred = rf_sent.predict(latest_feat_sent)[0]
    rf_sent_proba = max(rf_sent.predict_proba(latest_feat_sent)[0]) * 100
    sent_dir = "UP" if rf_sent_pred == 1 else "DOWN"
    
    # 3. LightGBM Model
    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(latest_feat)[0]
    lgb_proba = max(lgb_model.predict_proba(latest_feat)[0]) * 100
    lgb_dir = "UP" if lgb_pred == 1 else "DOWN"
    
    results = {
        "pure_tech": {"direction": pure_dir, "confidence": f"{rf_pure_proba:.1f}%"},
        "sentiment": {"direction": sent_dir, "confidence": f"{rf_sent_proba:.1f}%"},
        "lightgbm": {"direction": lgb_dir, "confidence": f"{lgb_proba:.1f}%"},
        "consensus": (pure_dir == sent_dir == lgb_dir)
    }
    
    return results, df['Close'].iloc[-1], sentiment_score, None

# --- DATABASE LOGGING & GRADING ---
def log_prediction_to_supabase(ticker, results):
    today_str = str(datetime.today().date())
    existing = supabase.table("forward_predictions").select("*").eq("date", today_str).eq("ticker", ticker).execute()
    
    if not existing.data:
        payload = {
            "date": today_str,
            "ticker": ticker,
            "confidence": results["pure_tech"]["confidence"],
            "pure_tech_forecast": results["pure_tech"]["direction"],
            "sentiment_forecast": results["sentiment"]["direction"],
            "lightgbm_forecast": results["lightgbm"]["direction"],
            "consensus_reached": results["consensus"],
            "status": "Pending"
        }
        try:
            supabase.table("forward_predictions").insert(payload).execute()
        except Exception as e:
            st.error(f"Supabase Insert Error: {e}")

def update_pending_predictions():
    try:
        response = supabase.table("forward_predictions").select("*").eq("status", "Pending").execute()
        rows = response.data
        
        for row in rows:
            pred_date = row["date"]
            ticker = row["ticker"]
            
            if pred_date < str(datetime.today().date()):
                historical_df = yf.download(ticker, start=pred_date, progress=False)
                if isinstance(historical_df.columns, pd.MultiIndex):
                    historical_df.columns = historical_df.columns.get_level_values(0)
                
                if len(historical_df) >= 2:
                    start_price = historical_df['Close'].iloc[0]
                    end_price = historical_df['Close'].iloc[-1]
                    actual_dir = "UP" if end_price > start_price else "DOWN"
                    
                    is_correct = row.get("pure_tech_forecast") == actual_dir
                    new_status = "✅ Correct" if is_correct else "❌ Incorrect"
                    
                    supabase.table("forward_predictions").update({"status": new_status}).eq("id", row["id"]).execute()
    except Exception:
        pass

# --- STREAMLIT UI TABS ---
st.title("📈 Market AI Multi-Model Companion v10")
st.markdown("Autonomous multi-model consensus engine, cloud cron sync, live scorecard analytics, and real-time consensus scanning.")

update_pending_predictions()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Multi-Model Analyzer", 
    "📂 Asset Categories", 
    "📋 Supabase Logs", 
    "📊 Model Scorecard",
    "🔥 Consensus Scanner"
])

with tab1:
    st.subheader("Live Multi-Model Inference")
    
    ticker_options = []
    ticker_mapping = {}
    for category, assets in ASSET_DICTIONARY.items():
        for symbol, name in assets.items():
            display_str = f"[{symbol}] {name}"
            ticker_options.append(display_str)
            ticker_mapping[display_str] = symbol
            
    selected_display = st.selectbox("Select Ticker Symbol & Asset", ticker_options)
    ticker_input = ticker_mapping[selected_display]

    if st.button("Run Models & Log Consensus", key="run_tab1"):
        with st.spinner(f"Analyzing {ticker_input} across Pure Tech, Sentiment, and LightGBM..."):
            results, current_price, sentiment, err = run_models(ticker_input)
            
            if err:
                st.error(err)
            else:
                st.metric("Current Price", f"${current_price:.2f}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pure Tech (RF)", results["pure_tech"]["direction"], results["pure_tech"]["confidence"])
                with col2:
                    st.metric("Sentiment (RF)", results["sentiment"]["direction"], results["sentiment"]["confidence"])
                with col3:
                    st.metric("LightGBM", results["lightgbm"]["direction"], results["lightgbm"]["confidence"])
                
                if results["consensus"]:
                    st.success(f"🔥 **THREE-WAY CONSENSUS REACHED:** All models agree on **{results['pure_tech']['direction']}** for {ticker_input}!")
                else:
                    st.info("ℹ️ Models diverge. No full consensus today.")
                
                log_prediction_to_supabase(ticker_input, results)
                st.toast("Prediction logged to Supabase successfully!", icon="🚀")

with tab2:
    st.subheader("📂 Dual-Dropdown Asset Categories & Portfolio Scanner")
    
    category_names = list(ASSET_DICTIONARY.keys())
    selected_category = st.selectbox("Select Investment Group / Category", category_names)
    
    category_assets = ASSET_DICTIONARY[selected_category]
    asset_display_options = [f"[{sym}] {nm}" for sym, nm in category_assets.items()]
    
    selected_asset_display = st.selectbox("Select Asset from Category", asset_display_options)
    selected_asset = [sym for sym in category_assets.keys() if f"[{sym}]" in selected_asset_display][0]
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"**Active Category:** {selected_category}")
    with col_info2:
        st.info(f"**Target Asset:** {selected_asset}")
        
    if st.button(f"Analyze {selected_asset} Now", key="run_tab2"):
        with st.spinner(f"Running models for {selected_asset}..."):
            results, current_price, sentiment, err = run_models(selected_asset)
            if err:
                st.error(err)
            else:
                st.metric("Current Price", f"${current_price:.2f}")
                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Pure Tech", results["pure_tech"]["direction"], results["pure_tech"]["confidence"])
                with colB:
                    st.metric("Sentiment", results["sentiment"]["direction"], results["sentiment"]["confidence"])
                with colC:
                    st.metric("LightGBM", results["lightgbm"]["direction"], results["lightgbm"]["confidence"])
                
                if results["consensus"]:
                    st.success(f"🔥 **CONSENSUS REACHED:** All 3 models agree on **{results['pure_tech']['direction']}** for {selected_asset}!")
                else:
                    st.warning("⚠️ Mixed signals across models.")
                
                log_prediction_to_supabase(selected_asset, results)
                st.toast("Prediction logged to Supabase successfully!", icon="🚀")

with tab3:
    st.subheader("📋 Supabase Prediction & Consensus Logs")
    
    filter_consensus = st.checkbox("Show Three-Way Consensus Only (consensus_reached = true)")
    
    query = supabase.table("forward_predictions").select("*").order("date", desc=True)
    if filter_consensus:
        query = query.eq("consensus_reached", True)
        
    logs_response = query.limit(50).execute()
    logs_data = logs_response.data
    
    if logs_data:
        logs_df = pd.DataFrame(logs_data)
        st.dataframe(logs_df, use_container_width=True)
    else:
        st.info("No matching prediction logs found in Supabase.")

with tab4:
    st.subheader("📊 Model Scorecard & Win Rate Trends")
    
    response = supabase.table("forward_predictions").select("*").order("date", desc=True).execute()
    data = response.data
    
    if data:
        df_score = pd.DataFrame(data)
        today_str = str(datetime.today().date())
        past_rows = df_score[df_score["date"] < today_str].copy()
        
        if not past_rows.empty:
            evaluated_records = []
            
            for _, row in past_rows.head(50).iterrows():
                pred_date = row["date"]
                ticker = row["ticker"]
                
                hist = yf.download(ticker, start=pred_date, progress=False)
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                    
                if len(hist) >= 2:
                    start_p = hist['Close'].iloc[0]
                    end_p = hist['Close'].iloc[-1]
                    actual_dir = "UP" if end_p > start_p else "DOWN"
                    
                    p_match = row.get("pure_tech_forecast") == actual_dir
                    s_match = row.get("sentiment_forecast") == actual_dir
                    l_match = row.get("lightgbm_forecast") == actual_dir
                    
                    evaluated_records.append({
                        "Date": pred_date,
                        "Ticker": ticker,
                        "Pure_Correct": 1 if p_match else 0,
                        "Sent_Correct": 1 if s_match else 0,
                        "LGB_Correct": 1 if l_match else 0,
                        "Pure Tech": "✅" if p_match else "❌",
                        "Sentiment": "✅" if s_match else "❌",
                        "LightGBM": "✅" if l_match else "❌",
                        "Consensus": "🔥 Match" if row.get("consensus_reached") else "Diverge"
                    })
            
            if evaluated_records:
                eval_df = pd.DataFrame(evaluated_records)
                eval_df = eval_df.sort_values("Date").reset_index(drop=True)
                
                total_evaluated = len(eval_df)
                pure_correct_total = eval_df["Pure_Correct"].sum()
                sent_correct_total = eval_df["Sent_Correct"].sum()
                lgb_correct_total = eval_df["LGB_Correct"].sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pure Tech (RF)", f"{(pure_correct_total/total_evaluated)*100:.1f}%")
                with col2:
                    st.metric("Sentiment (RF)", f"{(sent_correct_total/total_evaluated)*100:.1f}%")
                with col3:
                    st.metric("LightGBM", f"{(lgb_correct_total/total_evaluated)*100:.1f}%")
                    
                st.markdown("---")
                st.subheader("📈 Cumulative Win Rate Trend Over Time")
                
                eval_df["Pure Tech Trend"] = (eval_df["Pure_Correct"].expanding().mean()) * 100
                eval_df["Sentiment Trend"] = (eval_df["Sent_Correct"].expanding().mean()) * 100
                eval_df["LightGBM Trend"] = (eval_df["LGB_Correct"].expanding().mean()) * 100
                
                trend_chart_data = eval_df.set_index("Date")[["Pure Tech Trend", "Sentiment Trend", "LightGBM Trend"]]
                st.line_chart(trend_chart_data)
                
                st.markdown("---")
                st.subheader("📋 Model Breakdown Ledger")
                st.dataframe(eval_df[["Date", "Ticker", "Pure Tech", "Sentiment", "LightGBM", "Consensus"]], use_container_width=True)
            else:
                st.info("Past predictions exist, but historical price verification is still pending.")
        else:
            st.info("Predictions are logged daily, but none have reached a past date to be graded yet.")
    else:
        st.info("No prediction data found in Supabase yet.")

with tab5:
    st.subheader("🔥 Live 3-Way Consensus Scanner")
    st.markdown("Scan across an entire asset category to instantly find where all three models (Pure Tech, Sentiment, LightGBM) agree.")
    
    scan_category = st.selectbox("Select Category to Scan", list(ASSET_DICTIONARY.keys()), key="scan_category_sel")
    
    if st.button("🚀 Scan Category for Consensus", key="run_consensus_scan"):
        assets_dict = ASSET_DICTIONARY[scan_category]
        consensus_found = []
        
        progress_bar = st.progress(0)
        status_box = st.empty()
        
        items = list(assets_dict.items())
        total_items = len(items)
        
        for i, (symbol, name) in enumerate(items):
            status_box.text(f"Scanning [{symbol}] {name} ({i+1}/{total_items})...")
            results, current_price, sentiment, err = run_models(symbol)
            
            if not err and results and results["consensus"]:
                consensus_found.append({
                    "Ticker": symbol,
                    "Asset Name": name,
                    "Direction": results["pure_tech"]["direction"],
                    "Confidence": results["pure_tech"]["confidence"],
                    "Current Price": f"${current_price:.2f}"
                })
            progress_bar.progress((i + 1) / total_items)
            
        status_box.text("Scan complete!")
        progress_bar.empty()
        
        if consensus_found:
            st.success(f"🔥 Found **{len(consensus_found)}** strong 3-way consensus pick(s) in {scan_category}!")
            scan_df = pd.DataFrame(consensus_found)
            st.dataframe(scan_df, use_container_width=True)
        else:
            st.info(f"No 3-way consensus found across {scan_category} today. Models are currently mixed/diverging.")
