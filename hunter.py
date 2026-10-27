import os
import re
import time
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import numpy as np
import pandas as pd

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime.now(TZ)
TODAY = NOW.strftime("%Y-%m-%d")
TODAY_COMPACT = NOW.strftime("%Y%m%d")

def log(message):
    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {message}", flush=True)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    df = df.copy()
    close_col = "close" if "close" in df.columns else "收盘"
    
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[close_col]).reset_index(drop=True)

    if len(df) < 20: return None
    
    close = df[close_col]
    df["MA10"] = close.rolling(10).mean()
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    
    df["RSI"] = calculate_rsi(close)
    return df

def process_single_stock(row, today_compact):
    raw_code = str(row["代码"])
    code = re.sub(r'\D', '', raw_code).zfill(6)
    name = str(row["名称"])
    
    try:
        symbol_tx = f"sh{code}" if code.startswith('6') else f"sz{code}"
        hist = ak.stock_zh_a_hist_tx(
            symbol=symbol_tx, 
            start_date=(NOW - timedelta(days=40)).strftime("%Y%m%d"), 
            end_date=today_compact, 
            adjust="qfq"
        )

        if hist is None or len(hist) < 20: return None
            
        hist = calculate_indicators(hist)
        if hist is None: return None
            
        last = hist.iloc[-1]
        prev = hist.iloc[-2]
        
        ma10 = float(last["MA10"])
        macd = float(last["MACD"])
        prev_macd = float(prev["MACD"])
        rsi = float(last["RSI"])
        
        price = float(row["最新价"])
        pct = float(row["涨跌幅"])
        volume_ratio = float(row["量比"]) if "量比" in row and not pd.isna(row["量比"]) else 0
        weibi = float(row["委比"]) if "委比" in row and not pd.isna(row["委比"]) else 0
        outer = float(row["外盘"]) if "外盘" in row and not pd.isna(row["外盘"]) else 0
        inner = float(row["内盘"]) if "内盘" in row and not pd.isna(row["内盘"]) else 0
        
        # ================= 10分制评分系统 =================
        score = 0.0
        if price > ma10: score += 2
        if macd > prev_macd: score += 1
        if macd > 0: score += 1
        if 50 <= rsi <= 75: score += 1.5
        if 1.2 <= volume_ratio <= 3.0: score += 1
        if weibi > 10: score += 0.5
        if outer > inner: score += 1
        if 1.0 <= pct <= 4.5: score += 2
            
        if score < 7.0 or price <= ma10: return None

        target_price = round(price * 1.015, 2) 
        stop_loss = round(price * 0.98, 2)     

        return {
            "代码": code, "名称": name, "最新价": price,
            "涨幅%": round(pct, 2), "量比": round(volume_ratio, 2),
            "委比": round(weibi, 1), "外/内": round(outer/inner, 2) if inner>0 else 99,
            "RSI": round(rsi, 1), "评分": round(score, 1),
            "次日止盈": target_price, "止损": stop_loss
        }

    except Exception:
        return None

def scan_market():
    log("获取全市场实时快照...")
    try:
        spot = ak.stock_zh_a_spot_em()
    except Exception:
        spot = ak.stock_zh_a_spot().rename(columns={'代码': '代码', '名称': '名称', '最新价': '最新价', '涨跌幅': '涨跌幅', '成交额': '成交额', '成交量': '成交量'})
        for col in ["换手率", "量比", "委比", "外盘", "内盘"]:
            if col not in spot.columns: spot[col] = np.nan

    spot = spot.dropna(subset=["最新价", "涨跌幅", "成交额"])
    spot["纯代码"] = spot["代码"].astype(str).str.extract(r'(\d+)')[0].fillna('')
    
    # 涨幅条件限制
    condition_pct = np.where(
        spot["纯代码"].str.startswith('3'),
        spot["涨跌幅"] < 19.5,  
        spot["涨跌幅"] < 9.5    
    )
    
    # 【修改核心】允许 ST 股票买入，只过滤带有“退”字的退市股
    spot = spot[
        (~spot["名称"].str.contains("退", na=False)) # 允许 ST，但不碰退市
        & (~spot["纯代码"].str.startswith(('8', '4', '9', '688')))
        & (spot["最新价"] > 2) 
        & (spot["最新价"] <= 1000)       
        & (spot["成交额"] >= 20000000) 
        & condition_pct                  
    ].copy()

    log(f"基础过滤完毕(含ST股)，剩余 {len(spot)} 只股票进行全量深度计算...")
    log("⚠️ 并发已调至20，全量扫描耗时较长，请耐心等待...")

    candidates = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single_stock, row, TODAY_COMPACT): row for _, row in spot.iterrows()}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res: 
                candidates.append(res)
            if completed % 300 == 0:
                log(f"进度：{completed}/{len(spot)}，已发现 {len(candidates)} 支达标股")

    if not candidates:
        raise RuntimeError("今日全市场无符合 7.0 分标准的标的。")

    df_all = pd.DataFrame(candidates)
    df_top = df_all.sort_values(["评分", "量比"], ascending=[False, False])
    result = df_top.head(3) 
    
    os.makedirs("reports", exist_ok=True)
    result.to_csv(f"reports/{TODAY}.csv", index=False, encoding="utf-8-sig")
    
    report = [f"# 尾盘狙击计划：{TODAY}\n", f"- 达标标的：{len(result)} 只\n"]
    for i, item in enumerate(result.to_dict("records"), start=1):
        report.append(f"### {i}. {item['名称']}（{item['代码']}） | 评分: {item['评分']}")
        report.append(f"- **当前价**：{item['最新价']} (涨幅 {item['涨幅%']}%)")
        report.append(f"- 🎯 **次日止盈**：**{item['次日止盈']}** (+1.5% 离场)")
        report.append(f"- 🛑 **严格止损**：**{item['止损']}** (-2.0% 破位走人)")
        report.append(f"- 📊 盘口数据：量比 {item['量比']} | 委比 {item['委比']}% | 外/内 {item['外/内']} | RSI {item['RSI']}\n")
        
    with open("reports/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log("运行完成，请查看报告。")

if __name__ == "__main__":
    try:
        scan_market()
    except Exception:
        traceback.print_exc()
