# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def calculate_indicators(history: pd.DataFrame, config: dict) -> dict:
    if history is None or history.empty: raise ValueError("历史K线为空")
    df = history.copy()
    for column in ["open", "close", "high", "low", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["open", "close", "high", "low", "volume"]).sort_values("date").reset_index(drop=True)

    close = df["close"]
    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = (df["dif"] - df["dea"]) * 2
    
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma5"].replace(0, np.nan)

    latest = df.iloc[-1]
    return {
        "close": float(latest["close"]), "ma5": float(latest["ma5"]) if pd.notna(latest["ma5"]) else None,
        "ma10": float(latest["ma10"]) if pd.notna(latest["ma10"]) else None, "ma20": float(latest["ma20"]) if pd.notna(latest["ma20"]) else None,
        "macd": float(latest["macd"]) if pd.notna(latest["macd"]) else None, "rsi": float(latest["rsi"]) if pd.notna(latest["rsi"]) else None,
        "volume_ratio": float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else None,
        "above_ma20": bool(latest["close"] >= latest["ma20"]) if pd.notna(latest["ma20"]) else False,
        "macd_positive": bool(latest["macd"] > 0) if pd.notna(latest["macd"]) else False
    }
