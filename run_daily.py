import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from supabase import create_client

# --- SUPABASE SETUP (FROM SECRETS/ENV) ---
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# --- ASSET UNIVERSE ---
ASSET_DICTIONARY = {
    "🪙 Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "🥇 Commodities": ["GLD", "SLV", "USO"],
    "🔥 Growth": ["TQQQ", "SOXL", "NVDA", "TSLA"],
    "🛡️ Broad Market": ["SPY", "QQQ", "VTI"],
    "🌍 Global": ["^DJI", "^FTSE", "^GSPC"]
}

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
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

def get_news_sentiment(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news
        if not news:
            return 0.0
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(item.get('title', ''))['compound'] for item in news[:5]]
        return np.mean(scores) if scores else 0.0
    except Exception:
        return 0.0

def run_daily_prediction_for(ticker):
    df = fetch_stock_data(ticker)
    if len(df) < 60:
        return
    
    df = add_technical_indicators(df)
    feature_cols = ['SMA_10', 'SMA_50', 'RSI', 'Volatility']
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    model_df = df.dropna(subset=feature_cols + ['Target'])
    
    X = model_df[feature_cols]
    y = model_df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    latest_feat = X.tail(1)
    
    # Models
    rf_pure = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_pure.fit(X_train, y_train)
    pure_dir = "UP" if rf_pure.predict(latest_feat)[0] == 1 else "DOWN"
    pure_conf = f"{max(rf_pure.predict_proba(latest_feat)[0]) * 100:.1f}%"
    
    sentiment_score = get_news_sentiment(ticker)
    X_train_sent = X_train.copy()
    X_train_sent['Sentiment'] = sentiment_score
    latest_feat_sent = latest_feat.copy()
    latest_feat_sent['Sentiment'] = sentiment_score
    
    rf_sent = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_sent.fit(X_train_sent, y_train)
    sent_dir = "UP" if rf_sent.predict(latest_feat_sent)[0] == 1 else "DOWN"
    
    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_dir = "UP" if lgb_model.predict(latest_feat)[0] == 1 else "DOWN"
    
    consensus = (pure_dir == sent_dir == lgb_dir)
    
    # Log to Supabase
    today_str = str(datetime.today().date())
    existing = supabase.table("forward_predictions").select("*").eq("date", today_str).eq("ticker", ticker).execute()
    
    if not existing.data:
        payload = {
            "date": today_str,
            "ticker": ticker,
            "confidence": pure_conf,
            "pure_tech_forecast": pure_dir,
            "sentiment_forecast": sent_dir,
            "lightgbm_forecast": lgb_dir,
            "consensus_reached": consensus,
            "status": "Pending"
        }
        supabase.table("forward_predictions").insert(payload).execute()
        print(f"Logged {ticker}: Consensus={consensus}")

if __name__ == "__main__":
    print("Starting daily multi-model inference batch...")
    for category, tickers in ASSET_DICTIONARY.items():
        for t in tickers:
            try:
                print(f"Running {t}...")
                run_daily_prediction_for(t)
            except Exception as e:
                print(f"Error running {t}: {e}")
    print("Daily batch complete!")
