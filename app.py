import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from supabase import create_client

# --- PAGE CONFIG & MOBILE RESPONSIVENESS ---
st.set_page_config(
    page_title="Market AI Companion v0.1a14",
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

TEN_YEAR_ELIGIBLE_CATEGORIES = {
    "🥇 Commodities & Safe Havens",
    "🛡️ Stable Broad Market",
    "📊 Top 5 Performing ETFs",
    "💼 iShares Moderate / Core",
    "🌍 Global Indexes"
}

TEN_YEAR_ELIGIBLE_TICKERS = {
    sym
    for cat, assets in ASSET_DICTIONARY.items()
    if cat in TEN_YEAR_ELIGIBLE_CATEGORIES
    for sym in assets
}

def get_lookback_options(category=None, ticker=None):
    if category is not None:
        allows_10y = category in TEN_YEAR_ELIGIBLE_CATEGORIES
    elif ticker is not None:
        allows_10y = ticker in TEN_YEAR_ELIGIBLE_TICKERS
    else:
        allows_10y = False
    return ["3Y", "5Y", "10Y"] if allows_10y else ["3Y", "5Y"]

def render_lookback_selector(label, options, key):
    default_idx = options.index("5Y") if "5Y" in options else 0
    selected = st.selectbox(label, options, index=default_idx, key=key)
    return int(selected.replace("Y", ""))

# --- HELPER FUNCTIONS ---
def _extract_1d_series(df, col_name):
    series = df[col_name]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series.astype(float)

@st.cache_data(ttl=3600)
def fetch_macro_series(start_str, end_str):
    macro = {}
    for sym in ["^VIX", "^TNX"]:
        try:
            m_df = yf.download(sym, start=start_str, end=end_str, progress=False)
            if isinstance(m_df.columns, pd.MultiIndex):
                m_df.columns = m_df.columns.get_level_values(0)
            if not m_df.empty and "Close" in m_df.columns:
                macro[sym] = _extract_1d_series(m_df, "Close")
        except Exception:
            pass
    return macro

def fetch_stock_data(ticker, years=5):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years + 45)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    out = df.copy()
    close = _extract_1d_series(out, "Close")
    high = _extract_1d_series(out, "High") if "High" in out.columns else close
    low = _extract_1d_series(out, "Low") if "Low" in out.columns else close

    sma_10 = close.rolling(window=10).mean()
    sma_50 = close.rolling(window=50).mean()
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()

    out["Close_Clean"] = close
    out["SMA_10"] = sma_10
    out["SMA_50"] = sma_50
    out["Price_to_SMA10"] = (close / sma_10.replace(0, np.nan)) - 1.0
    out["Price_to_SMA50"] = (close / sma_50.replace(0, np.nan)) - 1.0
    out["RSI"] = compute_rsi(close, 14)
    out["MACD_Hist"] = macd_line - macd_signal
    out["BB_Width"] = (4.0 * std_20) / sma_20.replace(0, np.nan)
    out["ATR_Pct"] = atr_14 / close.replace(0, np.nan)
    out["Volatility"] = close.pct_change().rolling(window=10).std()
    out["Ret_1d"] = close.pct_change(1)
    out["Ret_3d"] = close.pct_change(3)
    out["Ret_5d"] = close.pct_change(5)

    if not out.empty:
        start_str = str(out.index.min().date())
        end_str = str((out.index.max() + timedelta(days=1)).date())
        macro = fetch_macro_series(start_str, end_str)
        vix_s = macro.get("^VIX")
        tnx_s = macro.get("^TNX")
        out["VIX_Level"] = vix_s.reindex(out.index).ffill().bfill() if vix_s is not None else 0.0
        out["TNX_Level"] = tnx_s.reindex(out.index).ffill().bfill() if tnx_s is not None else 0.0
    else:
        out["VIX_Level"] = 0.0
        out["TNX_Level"] = 0.0

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(inplace=True)
    return out

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

def _build_rf():
    return RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=42)

def _build_lgb():
    return lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=23,
        min_child_samples=15,
        random_state=42,
        verbose=-1
    )

def _build_meta_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=500, random_state=42)
    )

def walk_forward_and_stack(X_hist, y_hist, sentiment_score, latest_prob_row, n_splits=5):
    """
    Runs walk-forward validation across sequential time folds, collects out-of-fold
    upward probabilities from all 3 base models, and trains a stacking meta-classifier
    with volatility regime context (VIX_Level, ATR_Pct).
    """
    if len(X_hist) < (n_splits + 1) * 20:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    pure_scores, sent_scores, lgb_scores = [], [], []
    oof_rows = []
    oof_targets = []

    X_sent_hist = X_hist.copy()
    X_sent_hist["Sentiment"] = sentiment_score

    for train_idx, val_idx in tscv.split(X_hist):
        X_tr, y_tr = X_hist.iloc[train_idx], y_hist.iloc[train_idx]
        X_va, y_va = X_hist.iloc[val_idx], y_hist.iloc[val_idx]

        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            continue

        rf_p = _build_rf().fit(X_tr, y_tr)
        p_pure = rf_p.predict_proba(X_va)[:, 1]
        pure_scores.append(accuracy_score(y_va, (p_pure >= 0.5).astype(int)))

        rf_s = _build_rf().fit(X_sent_hist.iloc[train_idx], y_tr)
        p_sent = rf_s.predict_proba(X_sent_hist.iloc[val_idx])[:, 1]
        sent_scores.append(accuracy_score(y_va, (p_sent >= 0.5).astype(int)))

        lgb_m = _build_lgb().fit(X_tr, y_tr)
        p_lgb = lgb_m.predict_proba(X_va)[:, 1]
        lgb_scores.append(accuracy_score(y_va, (p_lgb >= 0.5).astype(int)))

        fold_meta = pd.DataFrame({
            "p_pure": p_pure,
            "p_sent": p_sent,
            "p_lgb": p_lgb,
            "VIX_Level": X_va["VIX_Level"].values,
            "ATR_Pct": X_va["ATR_Pct"].values
        })
        oof_rows.append(fold_meta)
        oof_targets.append(y_va.values)

    if not oof_rows:
        return None

    X_meta = pd.concat(oof_rows, ignore_index=True)
    y_meta = np.concatenate(oof_targets)

    meta_wf_acc = None
    if len(X_meta) >= 60 and len(np.unique(y_meta)) == 2:
        meta_tscv = TimeSeriesSplit(n_splits=3)
        m_accs = []
        for m_tr, m_va in meta_tscv.split(X_meta):
            if len(np.unique(y_meta[m_tr])) < 2 or len(np.unique(y_meta[m_va])) < 2:
                continue
            m_clf = _build_meta_model().fit(X_meta.iloc[m_tr], y_meta[m_tr])
            m_accs.append(accuracy_score(y_meta[m_va], m_clf.predict(X_meta.iloc[m_va])))
        if m_accs:
            meta_wf_acc = float(np.mean(m_accs) * 100)

    if len(np.unique(y_meta)) == 2 and len(X_meta) >= 30:
        meta_final = _build_meta_model().fit(X_meta, y_meta)
        meta_proba = meta_final.predict_proba(latest_prob_row)[0]
        meta_up_prob = float(meta_proba[1])
    else:
        meta_up_prob = float(np.mean([
            latest_prob_row["p_pure"].iloc[0],
            latest_prob_row["p_sent"].iloc[0],
            latest_prob_row["p_lgb"].iloc[0]
        ]))

    meta_dir = "UP" if meta_up_prob >= 0.5 else "DOWN"
    meta_conf = f"{max(meta_up_prob, 1.0 - meta_up_prob) * 100:.1f}%"

    return {
        "pure_wf": float(np.mean(pure_scores) * 100) if pure_scores else None,
        "sent_wf": float(np.mean(sent_scores) * 100) if sent_scores else None,
        "lgb_wf": float(np.mean(lgb_scores) * 100) if lgb_scores else None,
        "meta_dir": meta_dir,
        "meta_conf": meta_conf,
        "meta_wf": meta_wf_acc
    }

# --- MODEL TRAINING & PREDICTION ---
def run_models(ticker, years=5):
    df = fetch_stock_data(ticker, years=years)
    if len(df) < 80:
        return None, None, None, "Insufficient data (need at least 80 days of history)."

    df = add_technical_indicators(df)
    feature_cols = [
        "Price_to_SMA10",
        "Price_to_SMA50",
        "RSI",
        "MACD_Hist",
        "BB_Width",
        "ATR_Pct",
        "Volatility",
        "Ret_1d",
        "Ret_3d",
        "Ret_5d",
        "VIX_Level",
        "TNX_Level"
    ]

    df["Target"] = (df["Close_Clean"].shift(-1) > df["Close_Clean"]).astype(int)
    labeled_df = df.dropna(subset=feature_cols + ["Target"])
    latest_feat = df[feature_cols].tail(1)

    if len(labeled_df) < 50 or latest_feat.empty:
        return None, None, None, "Insufficient valid rows after feature engineering."

    X_hist = labeled_df[feature_cols]
    y_hist = labeled_df["Target"]
    sentiment_score = get_news_sentiment(ticker)

    # 1. Pure Technical Random Forest
    rf_pure = _build_rf().fit(X_hist, y_hist)
    pure_up_prob = float(rf_pure.predict_proba(latest_feat)[0][1])
    pure_dir = "UP" if pure_up_prob >= 0.5 else "DOWN"
    pure_conf = f"{max(pure_up_prob, 1.0 - pure_up_prob) * 100:.1f}%"

    # 2. Sentiment-Enhanced Random Forest (FinBERT)
    X_sent_hist = X_hist.copy()
    X_sent_hist["Sentiment"] = sentiment_score
    latest_feat_sent = latest_feat.copy()
    latest_feat_sent["Sentiment"] = sentiment_score

    rf_sent = _build_rf().fit(X_sent_hist, y_hist)
    sent_up_prob = float(rf_sent.predict_proba(latest_feat_sent)[0][1])
    sent_dir = "UP" if sent_up_prob >= 0.5 else "DOWN"
    sent_conf = f"{max(sent_up_prob, 1.0 - sent_up_prob) * 100:.1f}%"

    # 3. LightGBM Model
    lgb_model = _build_lgb().fit(X_hist, y_hist)
    lgb_up_prob = float(lgb_model.predict_proba(latest_feat)[0][1])
    lgb_dir = "UP" if lgb_up_prob >= 0.5 else "DOWN"
    lgb_conf = f"{max(lgb_up_prob, 1.0 - lgb_up_prob) * 100:.1f}%"

    # 4. Walk-Forward Validation + Stacking Meta-Model
    latest_prob_row = pd.DataFrame({
        "p_pure": [pure_up_prob],
        "p_sent": [sent_up_prob],
        "p_lgb": [lgb_up_prob],
        "VIX_Level": [float(latest_feat["VIX_Level"].iloc[0])],
        "ATR_Pct": [float(latest_feat["ATR_Pct"].iloc[0])]
    })

    stack_out = walk_forward_and_stack(
        X_hist,
        y_hist,
        sentiment_score,
        latest_prob_row,
        n_splits=5
    )

    if stack_out is not None:
        pure_wf = stack_out["pure_wf"]
        sent_wf = stack_out["sent_wf"]
        lgb_wf = stack_out["lgb_wf"]
        meta_dir = stack_out["meta_dir"]
        meta_conf = stack_out["meta_conf"]
        meta_wf = stack_out["meta_wf"]
    else:
        pure_wf = sent_wf = lgb_wf = meta_wf = None
        avg_prob = float(np.mean([pure_up_prob, sent_up_prob, lgb_up_prob]))
        meta_dir = "UP" if avg_prob >= 0.5 else "DOWN"
        meta_conf = f"{max(avg_prob, 1.0 - avg_prob) * 100:.1f}%"

    results = {
        "pure_tech": {"direction": pure_dir, "confidence": pure_conf, "wf_acc": pure_wf},
        "sentiment": {"direction": sent_dir, "confidence": sent_conf, "wf_acc": sent_wf},
        "lightgbm": {"direction": lgb_dir, "confidence": lgb_conf, "wf_acc": lgb_wf},
        "meta_ensemble": {"direction": meta_dir, "confidence": meta_conf, "wf_acc": meta_wf},
        "consensus": (pure_dir == sent_dir == lgb_dir)
    }

    current_price = float(df["Close_Clean"].iloc[-1])
    return results, current_price, sentiment_score, None

# --- DATABASE LOGGING & GRADING ---
def log_prediction_to_supabase(ticker, results):
    today_str = str(datetime.today().date())
    existing = supabase.table("forward_predictions").select("*").eq("date", today_str).eq("ticker", ticker).execute()

    if not existing.data:
        payload = {
            "date": today_str,
            "ticker": ticker,
            "confidence": results["meta_ensemble"]["confidence"],
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
                    close_series = _extract_1d_series(hist, "Close")
                    start_p = float(close_series.iloc[0])
                    end_p = float(close_series.iloc[-1])
                    actual_dir = "UP" if end_p > start_p else "DOWN"

                    is_correct = row.get("pure_tech_forecast") == actual_dir
                    new_status = "✅ Correct" if is_correct else "❌ Incorrect"

                    supabase.table("forward_predictions").update({"status": new_status}).eq("id", row["id"]).execute()
    except Exception:
        pass

def _format_wf_caption(model_res):
    wf = model_res.get("wf_acc")
    return f"Walk-Forward CV: {wf:.1f}%" if wf is not None else "Walk-Forward CV: N/A"

# --- STREAMLIT UI ---
st.title("📈 Market AI Companion v0.1a14")
st.markdown("Mobile-Optimized Stacking Ensemble, FinBERT & Walk-Forward Validation.")

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

    t1_options = get_lookback_options(ticker=ticker_input)
    years_t1 = render_lookback_selector("Historical Lookback", t1_options, key="lb_t1")
    if "10Y" not in t1_options:
        st.caption("10Y history is hidden for crypto and high-risk/growth assets.")

    if st.button("Run & Log Prediction", key="run_t1"):
        with st.spinner(f"Running {years_t1}Y walk-forward stacking ensemble..."):
            results, price, sentiment, err = run_models(ticker_input, years=years_t1)
            if err:
                st.error(err)
            else:
                st.metric("Current Price", f"${price:.2f}")
                st.metric("FinBERT Sentiment Score", f"{sentiment:+.3f}")

                st.metric(
                    "🧠 Meta-Ensemble (Stacker)",
                    results["meta_ensemble"]["direction"],
                    results["meta_ensemble"]["confidence"]
                )
                st.caption(_format_wf_caption(results["meta_ensemble"]))

                st.metric("Pure Tech (RF)", results["pure_tech"]["direction"], results["pure_tech"]["confidence"])
                st.caption(_format_wf_caption(results["pure_tech"]))

                st.metric("Sentiment (RF)", results["sentiment"]["direction"], results["sentiment"]["confidence"])
                st.caption(_format_wf_caption(results["sentiment"]))

                st.metric("LightGBM", results["lightgbm"]["direction"], results["lightgbm"]["confidence"])
                st.caption(_format_wf_caption(results["lightgbm"]))

                if results["consensus"]:
                    st.success(f"🔥 **3-WAY CONSENSUS:** {ticker_input} is **{results['pure_tech']['direction']}**!")
                else:
                    st.info(f"ℹ️ Base models diverge — Meta-Stacker leans **{results['meta_ensemble']['direction']}**.")
                log_prediction_to_supabase(ticker_input, results)
                st.toast("Logged to Supabase!", icon="🚀")

with tab2:
    st.subheader("Category Scanner")
    sel_cat = st.selectbox("Select Category", list(ASSET_DICTIONARY.keys()))
    cat_assets = ASSET_DICTIONARY[sel_cat]
    sel_disp = st.selectbox("Select Symbol", [f"[{s}] {n}" for s, n in cat_assets.items()])
    sel_sym = [s for s in cat_assets.keys() if f"[{s}]" in sel_disp][0]

    t2_options = get_lookback_options(category=sel_cat)
    years_t2 = render_lookback_selector("Historical Lookback", t2_options, key="lb_t2")
    if "10Y" not in t2_options:
        st.caption("10Y history is hidden for this category to avoid stale regime noise.")

    if st.button(f"Analyze {sel_sym}", key="run_t2"):
        with st.spinner(f"Analyzing {years_t2}Y stacking ensemble..."):
            results, price, sentiment, err = run_models(sel_sym, years=years_t2)
            if err:
                st.error(err)
            else:
                st.metric("Price", f"${price:.2f}")
                st.metric("FinBERT Sentiment Score", f"{sentiment:+.3f}")

                st.metric(
                    "🧠 Meta-Ensemble (Stacker)",
                    results["meta_ensemble"]["direction"],
                    results["meta_ensemble"]["confidence"]
                )
                st.caption(_format_wf_caption(results["meta_ensemble"]))

                st.metric("Pure Tech", results["pure_tech"]["direction"], results["pure_tech"]["confidence"])
                st.caption(_format_wf_caption(results["pure_tech"]))

                st.metric("Sentiment", results["sentiment"]["direction"], results["sentiment"]["confidence"])
                st.caption(_format_wf_caption(results["sentiment"]))

                st.metric("LightGBM", results["lightgbm"]["direction"], results["lightgbm"]["confidence"])
                st.caption(_format_wf_caption(results["lightgbm"]))

                if results["consensus"]:
                    st.success(f"🔥 **Consensus:** {sel_sym} is **{results['pure_tech']['direction']}**!")
                else:
                    st.warning(f"⚠️ Mixed signals — Meta-Stacker leans **{results['meta_ensemble']['direction']}**.")
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
                    close_series = _extract_1d_series(hist, "Close")
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
    t5_options = get_lookback_options(category=scan_cat)
    years_t5 = render_lookback_selector("Historical Lookback", t5_options, key="lb_t5")

    if st.button("🚀 Scan Category for Consensus", key="run_scan"):
        found = []
        bar = st.progress(0)
        items = list(ASSET_DICTIONARY[scan_cat].items())
        total_items = len(items)

        for i, (sym, name) in enumerate(items):
            res, pr, sent, err = run_models(sym, years=years_t5)
            if not err and res and res["consensus"]:
                found.append({
                    "Ticker": sym,
                    "Asset Name": name,
                    "Direction": res["pure_tech"]["direction"],
                    "Meta Stacker": f"{res['meta_ensemble']['direction']} ({res['meta_ensemble']['confidence']})",
                    "Meta CV": f"{res['meta_ensemble']['wf_acc']:.1f}%" if res['meta_ensemble']['wf_acc'] is not None else "N/A"
                })
            bar.progress((i + 1) / total_items)
        bar.empty()

        if found:
            st.success(f"🔥 Found **{len(found)}** consensus pick(s) in {scan_cat}!")
            st.dataframe(pd.DataFrame(found), use_container_width=True)
        else:
            st.info(f"No 3-way consensus found across {scan_cat} today.")
