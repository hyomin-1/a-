import os
import re
import time
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import akshare as ak
import numpy as np
import pandas as pd

# ================= 全局网络重试会话配置 =================
def create_retry_session(retries=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

GLOBAL_SESSION = create_retry_session()
# =========================================================

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime.now(TZ)
TODAY = NOW.strftime("%Y-%m-%d")
TODAY_COMPACT = NOW.strftime("%Y%m%d")

def log(message):
    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {message}", flush=True)

def calculate_indicators(df):
    df = df.copy()
    close_col = "收盘" if "收盘" in df.columns else "close"
    vol_col = "成交量" if "成交量" in df.columns else "volume"
    high_col = "最高" if "最高" in df.columns else "high"
    low_col = "最低" if "最低" in df.columns else "low"
    
    for c in [close_col, vol_col, high_col, low_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.dropna(subset=[close_col]).reset_index(drop=True)

    if len(df) < 30:
        return None
    
    close = df[close_col]
    high = df[high_col]
    low = df[low_col]
    
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    df["成交量均值5"] = df[vol_col].rolling(5).mean()
    
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df["ATR"] = tr.rolling(14).mean()
    
    return df

def process_single_stock(row, today_compact):
    raw_code = str(row["代码"])
    code = re.sub(r'\D', '', raw_code).zfill(6)
    name = str(row["名称"])
    
    try:
        hist = None
        try:
            hist = ak.stock_zh_a_hist(
                symbol=code, 
                period="daily", 
                start_date=(NOW - timedelta(days=100)).strftime("%Y%m%d"), 
                end_date=today_compact, 
                adjust="qfq"
            )
        except Exception:
            if code.startswith('6'): symbol_sina = f"sh{code}"
            elif code.startswith('0') or code.startswith('3'): symbol_sina = f"sz{code}"
            else: return None, f"{code} 不支持"
            
            # 新浪日线接口切换兜底
            hist = ak.stock_zh_a_daily(symbol=symbol_sina, adjust="qfq")
            hist = hist.rename(columns={'date': '日期', 'close': '收盘', 'volume': '成交量', 'high': '最高', 'low': '最低'})

        if hist is None or len(hist) < 30:
            return None, f"{code} 数据不足"
            
        last_date_str = str(hist.iloc[-1]["日期"]).replace("-", "")[:8]
        if last_date_str != today_compact:
            return None, f"{code} 数据未更新"

        hist = calculate_indicators(hist)
        if hist is None: return None, f"{code} 指标失败"
            
        last = hist.iloc[-1]
        prev = hist.iloc[-2]
        
        price = float(last["收盘"])
        pct = float(row["涨跌幅"])
        turnover = float(row["换手率"])
        amount = float(row["成交额"])
        ma5 = float(last["MA5"])
        ma10 = float(last["MA10"])
        ma20 = float(last["MA20"])
        macd = float(last["MACD"])
        prev_macd = float(prev["MACD"])
        vol = float(last["成交量"])
        vol5 = float(last["成交量均值5"])
        atr = float(last["ATR"])
        
        if not np.isfinite(atr) or atr <= 0: atr = price * 0.03

        score = 0.0
        reasons = []
        if price >= ma5: score += 1; reasons.append("站上MA5")
        if ma5 >= ma10: score += 1; reasons.append("短期均线偏强")
        if price >= ma20: score += 1; reasons.append("站上MA20")
        if macd > prev_macd: score += 1; reasons.append("MACD改善")
        if macd > 0: score += 0.5; reasons.append("MACD红柱")
        if vol5 > 0 and vol < vol5 * 1.5: score += 0.5; reasons.append("量能温和")
        if 0 <= pct <= 3: score += 1; reasons.append("涨幅适中")
        if 1 <= turnover <= 10: score += 0.5; reasons.append("换手活跃")
        if pct > 5: score -= 2; reasons.append("追高风险")

        target_price_1 = price + (atr * 1.5)
        stop_loss = min(ma5, price - (atr * 1.5))
        if stop_loss >= price: stop_loss = price * 0.97

        exp_profit_pct = ((target_price_1 - price) / price) * 100
        risk_pct = ((price - stop_loss) / price) * 100

        candidate = {
            "代码": code, "名称": name, "收盘价": round(price, 2),
            "当日涨跌幅%": round(pct, 2), "成交额(万)": int(amount/10000),
            "评分": round(score, 2),
            "止盈位": round(target_price_1, 2),
            "预期盈%": round(exp_profit_pct, 2),
            "止损位": round(stop_loss, 2),
            "风险%": round(risk_pct, 2),
            "理由": "；".join(reasons),
        }
        return candidate, None

    except Exception:
        return None, f"{code} 异常"

def scan_market():
    try:
        trade_dates = ak.tool_trade_date_hist_sina()
        trade_dates["trade_date"] = pd.to_datetime(trade_dates["trade_date"]).dt.strftime("%Y-%m-%d")
        if TODAY not in trade_dates["trade_date"].values:
            log(f"今天 {TODAY} 不是A股交易日，退出。")
            return
    except Exception as e:
        log(f"交易日历获取失败: {e}")

    spot = None
    try:
        log("正在获取沪深全市场实时行情...")
        spot = ak.stock_zh_a_spot_em()
    except Exception as e:
        log(f"东财实时接口异常({e})，尝试新浪快照兜底...")
        # 备用链路增加短暂休眠以缓解频控
        time.sleep(2) 
        try:
            spot = ak.stock_zh_a_spot().rename(columns={'代码': '代码', '名称': '名称', '最新价': '最新价', '涨跌幅': '涨跌幅', '成交额': '成交额'})
            if '换手率' not in spot.columns: spot['换手率'] = 1.0
        except Exception as e2:
            log(f"新浪快照接口同样超时: {e2}，执行最后一次等待重试...")
            time.sleep(5)
            spot = ak.stock_zh_a_spot_em()

    for col in ["最新价", "涨跌幅", "成交额", "换手率"]:
        spot[col] = pd.to_numeric(spot[col], errors="coerce")
    spot = spot.dropna(subset=["最新价", "涨跌幅", "成交额", "换手率"])
    
    spot["纯代码"] = spot["代码"].astype(str).str.extract(r'(\d+)')[0].fillna('')
    spot = spot[
        (~spot["名称"].str.contains("ST|退", na=False))
        & (~spot["纯代码"].str.startswith(('8', '4', '9', '688')))
        & (spot["最新价"] > 0) & (spot["成交额"] >= 3e7) 
        & (spot["换手率"] >= 0.5) & (spot["涨跌幅"] > -7) & (spot["涨跌幅"] < 7)
    ].copy()

    log(f"基础过滤后剩余 {len(spot)} 只，开始计算……")
    candidates = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_stock, row, TODAY_COMPACT): row for _, row in spot.iterrows()}
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            c, _ = future.result()
            if c: candidates.append(c)
            if completed_count % 100 == 0:
                log(f"进度：已扫描 {completed_count}/{len(spot)}，入选 {len(candidates)} 只")

    if not candidates:
        raise RuntimeError("没有取得有效候选数据。")

    df_all = pd.DataFrame(candidates)
    df_top = df_all[df_all["评分"] >= 4.0].sort_values(["评分", "成交额(万)"], ascending=[False, False])
    result = df_top.head(3)
    
    os.makedirs("reports", exist_ok=True)
    if len(result) > 0:
        result.to_csv(f"reports/{TODAY}.csv", index=False, encoding="utf-8-sig")
    
    report = [f"# A股猎手交易计划：{TODAY}\n", f"- 达标强信号股票：{len(result)} 只\n"]
    
    if len(result) == 0:
        report.append("> 今日无股票达到 4.0 分及格线，**建议空仓休息**。\n")
    else:
        for i, item in enumerate(result.to_dict("records"), start=1):
            report.append(f"### {i}. {item['名称']}（{item['代码']}） | 评分: {item['评分']}")
            report.append(f"- **收盘价**：{item['收盘价']} (当日涨幅 {item['当日涨跌幅%']}%)")
            report.append(f"- 🎯 **第一止盈位**：**{item['止盈位']}** (预期盈 **+{item['预期盈%']}%**)")
            report.append(f"- 🛑 **严格止损位**：**{item['止损位']}** (风险约 -{item['风险%']}%)")
            report.append(f"- 入选理由：{item['理由']}\n")
            
    with open("reports/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log("运行完成，请查看交易计划。")

if __name__ == "__main__":
    try:
        scan_market()
    except Exception:
        traceback.print_exc()
        raise
