import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from supabase import create_client

# --- PAGE CONFIG & MOBILE RESPONSIVENESS ---
st.set_page_config(
    page_title="Market AI Companion v10",
    page_icon="📈",
    layout="centered"
)

st.markdown("""
    <style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- SUPABASE SETUP ---
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# --- FINBERT SETUP ---
@st.cache_resource
def init_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# --- ASSET CATEGORIES & PORTFOLIOS ---
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
def fetch_stock_data(ticker, years=5):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)
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

        headlines = []
        for item in news[:5]:
            title = item.get("title") or item.get("content", {}).get("title", "")
            if title:
                headlines.append(title)

        if not headlines:
            return 0.0

        tokenizer, model = init_finbert()
        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        pos_idx = next((idx for idx, lbl in id2label.items() if "pos" in lbl), 0)
        neg_idx = next((idx for idx, lbl in id2label.items() if "neg" in lbl), 1)

        inputs = tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        scores = (probs[:, pos_idx] - probs[:, neg_idx]).cpu().numpy()
        return float(np.mean(scores)) if len(scores) > 0 else 0.0
    except Exception:
        return 0.0

# --- MODEL TRAINING & PREDICTION ---
def run_models(ticker, years=5):
    df = fetch_stock_data(ticker, years=years)
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
    pure_dir = "UP" if rf_pure.predict(latest_feat)[0] == 1 else "DOWN"
    pure_conf = f"{max(rf_pure.predict_proba(latest_feat)[0]) * 100:.1f}%"

    # 2. Sentiment-Enhanced Random Forest (FinBERT)
    sentiment_score = get_news_sentiment(ticker)
    X_train_sent = X_train.copy()
    latest_feat_sent = latest_feat.copy()
    X_train_sent['Sentiment'] = sentiment_score
    latest_feat_sent['Sentiment'] = sentiment_score

    rf_sent = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_sent.fit(X_train_sent, y_train)
    sent_dir = "UP" if rf_sent.predict(latest_feat_sent)[0] == 1 else "DOWN"
    sent_conf = f"{max(rf_sent.predict_proba(latest_feat_sent)[0]) * 100:.1f}%"

    # 3. LightGBM Model
    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_dir = "UP" if lgb_model.predict(latest_feat)[0] == 1 else "DOWN"
    lgb_conf = f"{max(lgb_model.predict_proba(latest_feat)[0]) * 100:.1f}%"

    results = {
        "pure_tech": {"direction": pure_dir, "confidence": pure_conf},
        "sentiment": {"direction": sent_dir, "confidence": sent_conf},
        "lightgbm": {"direction": lgb_dir, "confidence": lgb_conf},
        "consensus": (pure_dir == sent_dir == lgb_dir)
    }

    close_s = df['Close']
    if isinstance(close_s, pd.DataFrame):
        close_s = close_s.iloc[:, 0]
    current_price = float(close_s.iloc[-1])

    return results, current_price, sentiment_score, None

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
                hist = yf.download(ticker, start=pred_date, progress=False)
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)

                if len(hist) >= 2:
                    close_series = hist['Close']
                    if isinstance(close_series, pd.DataFrame):
                        close_series = close_series.iloc[:, 0]

                    start_p = float(close_series.iloc[0])
                    end_p = float(close_series.iloc[-1])
                    actual_dir = "UP" if end_p > start_p else "DOWN"

                    is_correct = row.get("pure_tech_forecast") == actual_dir
                    new_status = "✅ Correct" if is_correct else "❌ Incorrect"

                    supabase.table("forward_predictions").update({"status": new_status}).eq("id", row["id"]).execute()
    except Exception:
        pass

# --- STREAMLIT UI ---
st.title("📈 Market AI Companion v10")
st.markdown("Mobile-Optimized Multi-Model Consensus & Cloud Intelligence (FinBERT).")

update_pending_predictions()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Analyzer",
    "📂 Portfolios",
    "📋 Logs",
    "📊 Scorecard",
    "🔥 Scanner"
])

with tab1:
    st.subheader("Live Inference")
    ticker_options, ticker_mapping = [], {}
    for cat, assets in ASSET_DICTIONARY.items():
        for sym, name in assets.items():
            display_str = f"[{sym}] {name}"
            ticker_options.append(display_str)
            ticker_mapping[display_str] = sym

    selected_display = st.selectbox("Select Asset", ticker_options)
    ticker_input = ticker_mapping[selected_display]

    if st.button("Run & Log Prediction", key="run_t1"):
        with st.spinner("Running models with FinBERT..."):
            results, price, sentiment, err = run_models(ticker_input)
            if err:
                st.error(err)
            else:
                st.metric("Current Price", f"${price:.2f}")
                st.metric("FinBERT Sentiment Score", f"{sentiment:+.3f}")
                st.metric("Pure Tech (RF)", results["pure_tech"]["direction"], results["pure_tech"]["confidence"])
                st.metric("Sentiment (RF)", results["sentiment"]["direction"], results["sentiment"]["confidence"])
                st.metric("LightGBM", results["lightgbm"]["direction"], results["lightgbm"]["confidence"])

                if results["consensus"]:
                    st.success(f"🔥 **3-WAY CONSENSUS:** {ticker_input} is **{results['pure_tech']['direction']}**!")
                else:
                    st.info("ℹ️ Models diverge.")
                log_prediction_to_supabase(ticker_input, results)
                st.toast("Logged to Supabase!", icon="🚀")

with tab2:
    st.subheader("Category Scanner")
    sel_cat = st.selectbox("Select Category", list(ASSET_DICTIONARY.keys()))
    cat_assets = ASSET_DICTIONARY[sel_cat]
    sel_disp = st.selectbox("Select Symbol", [f"[{s}] {n}" for s, n in cat_assets.items()])
    sel_sym = [s for s in cat_assets.keys() if f"[{s}]" in sel_disp][0]

    if st.button(f"Analyze {sel_sym}", key="run_t2"):
        with st.spinner("Analyzing with FinBERT..."):
            results, price, sentiment, err = run_models(sel_sym)
            if err:
                st.error(err)
            else:
                st.metric("Price", f"${price:.2f}")
                st.metric("FinBERT Sentiment Score", f"{sentiment:+.3f}")
                st.metric("Pure Tech", results["pure_tech"]["direction"], results["pure_tech"]["confidence"])
                st.metric("Sentiment", results["sentiment"]["direction"], results["sentiment"]["confidence"])
                st.metric("LightGBM", results["lightgbm"]["direction"], results["lightgbm"]["confidence"])
                if results["consensus"]:
                    st.success(f"🔥 **Consensus:** {sel_sym} is **{results['pure_tech']['direction']}**!")
                else:
                    st.warning("⚠️ Mixed signals.")
                log_prediction_to_supabase(sel_sym, results)
                st.toast("Logged!", icon="🚀")

with tab3:
    st.subheader("Supabase Logs")
    only_consensus = st.checkbox("3-Way Consensus Only")
    query = supabase.table("forward_predictions").select("*").order("date", desc=True)
    if only_consensus:
        query = query.eq("consensus_reached", True)
    logs = query.limit(30).execute().data
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info("No logs found.")

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
                    close_series = hist['Close']
                    if isinstance(close_series, pd.DataFrame):
                        close_series = close_series.iloc[:, 0]

                    start_p = float(close_series.iloc[0])
                    end_p = float(close_series.iloc[-1])
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
    st.markdown("Scan across an entire asset category to instantly find where all three models agree.")

    scan_cat = st.selectbox("Select Category to Scan", list(ASSET_DICTIONARY.keys()), key="sc_cat")

    if st.button("🚀 Scan Category for Consensus", key="run_scan"):
        found = []
        bar = st.progress(0)
        items = list(ASSET_DICTIONARY[scan_cat].items())
        total_items = len(items)

        for i, (sym, name) in enumerate(items):
            res, pr, sent, err = run_models(sym)
            if not err and res and res["consensus"]:
                found.append({
                    "Ticker": sym,
                    "Asset Name": name,
                    "Direction": res["pure_tech"]["direction"],
                    "Confidence": res["pure_tech"]["confidence"]
                })
            bar.progress((i + 1) / total_items)
        bar.empty()

        if found:
            st.success(f"🔥 Found **{len(found)}** consensus pick(s) in {scan_cat}!")
            st.dataframe(pd.DataFrame(found), use_container_width=True)
        else:
            st.info(f"No 3-way consensus found across {scan_cat} today.")
