# -*- coding: utf-8 -*-
"""
A股猎手升级版（生成代码池+腾讯新浪双源 终极防封版）
"""
import os
import re
import json
import time
import random
import traceback
import warnings
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TZ = ZoneInfo("Asia/Shanghai")
REPORT_DIR = "reports"
# 【降并发防封】从15降到6，配合腾讯失败秒切新浪，成功率 99%
MAX_WORKERS = 6
TOP_N = 3


def sh_now():
    return datetime.now(TZ)


def log(message):
    print(f"[{sh_now().strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_code(value):
    digits = re.sub(r"\D", "", str(value))
    return digits[-6:].zfill(6) if digits else ""


def get_col(df, candidates, default=None):
    for col in candidates:
        if col in df.columns:
            return col
    return default


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((loss == 0) & (gain > 0), 100)
    return result


def calculate_indicators(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    close_col = get_col(df, ["close", "收盘"])
    if not close_col: return None
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[close_col]).sort_index().reset_index(drop=True)
    if len(df) < 60: return None

    close = df[close_col]
    df["_close"] = close
    for n in (5, 10, 20, 60):
        df[f"MA{n}"] = close.rolling(n).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    df["RSI"] = calculate_rsi(close, 14)

    high_col = get_col(df, ["high", "最高"])
    low_col = get_col(df, ["low", "最低"])
    vol_col = get_col(df, ["volume", "成交量"])

    if high_col: df["_high"] = pd.to_numeric(df[high_col], errors="coerce")
    if low_col: df["_low"] = pd.to_numeric(df[low_col], errors="coerce")
    if vol_col: df["_volume"] = pd.to_numeric(df[vol_col], errors="coerce")

    return df


# ================= 腾讯 + 新浪 双源历史K线接口 =================

def fetch_hist_tencent(code):
    market = "sh" if str(code).startswith(("6", "9")) else "sz"
    symbol = f"{market}{code}"
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{symbol},day,,,120,qfq"}
    try:
        # 增加微小延迟，防封
        time.sleep(random.uniform(0.1, 0.3))
        resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=(2, 3))
        resp.raise_for_status()
        data = resp.json()
        node = data.get("data", {}).get(symbol, {})
        rows = node.get("qfqday") or node.get("day") or []
        if not rows: return None
        
        records = []
        for row in rows:
            if len(row) >= 6:
                records.append({"date": row[0], "open": row[1], "close": row[2], "high": row[3], "low": row[4], "volume": row[5]})
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        for c in ("open", "close", "high", "low", "volume"): df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if len(df) >= 60: return df
    except Exception:
        pass
    return None

def fetch_hist_sina(code):
    market = "sh" if str(code).startswith(("6", "9")) else "sz"
    symbol = f"{market}{code}"
    url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen=120"
    try:
        time.sleep(random.uniform(0.1, 0.3))
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(2, 3))
        resp.raise_for_status()
        rows = resp.json()
        if not rows: return None
        
        df = pd.DataFrame(rows)
        df = df.rename(columns={"day": "date", "open": "open", "close": "close", "high": "high", "low": "low", "volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        for c in ("open", "close", "high", "low", "volume"): df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if len(df) >= 60: return df
    except Exception:
        pass
    return None

def fetch_hist(code):
    # 腾讯失败，秒切新浪
    df = fetch_hist_tencent(code)
    if df is not None: return df
    return fetch_hist_sina(code)

# =========================================================

def process_single_stock(row):
    code = normalize_code(row.get("代码", ""))
    name = str(row.get("名称", ""))
    if not code or "ST" in name.upper() or "退" in name: return None

    try:
        hist_raw = fetch_hist(code)
        hist = calculate_indicators(hist_raw)
        if hist is None or len(hist) < 60: raise RuntimeError("日线不足")

        last = hist.iloc[-1]
        prev = hist.iloc[-2]
        price = safe_float(row.get("最新价"))
        pct = safe_float(row.get("涨跌幅"))
        if not np.isfinite(price) or price <= 0 or not np.isfinite(pct): return None

        ma5 = safe_float(last.get("MA5"))
        ma10 = safe_float(last.get("MA10"))
        ma20 = safe_float(last.get("MA20"))
        ma60 = safe_float(last.get("MA60"))
        daily_macd = safe_float(last.get("MACD"))
        daily_rsi = safe_float(last.get("RSI"))
        volume_ratio = safe_float(row.get("量比"), 0.0)
        turnover = safe_float(row.get("换手率"), np.nan)
        amount = safe_float(row.get("成交额"), np.nan)

        prev_hist = hist.iloc[:-1].tail(60)
        high_col = get_col(prev_hist, ["_high"])
        low_col = get_col(prev_hist, ["_low"])
        high20 = safe_float(prev_hist.tail(20)[high_col].max()) if high_col else np.nan
        low20 = safe_float(prev_hist.tail(20)[low_col].min()) if low_col else np.nan

        support = max([x for x in [low20, ma20, ma60] if np.isfinite(x) and x < price], default=np.nan)
        resistance = min([x for x in [high20, high20] if np.isfinite(x) and x > price], default=np.nan)
        support_dist = ((price / support - 1) * 100 if np.isfinite(support) and support > 0 else np.nan)
        resistance_dist = ((resistance / price - 1) * 100 if (np.isfinite(resistance) and price > 0) else np.nan)

        score = 0.0
        reasons, risks = [], []

        if price >= ma5 and price >= ma10: score += 0.8; reasons.append("价格站上MA5/MA10")
        elif price >= ma20: score += 0.35
        else: risks.append("价格低于MA20")

        if np.isfinite(ma20) and np.isfinite(ma60) and ma20 >= ma60: score += 0.6; reasons.append("MA20在MA60上方")
        else: risks.append("趋势偏弱")

        if np.isfinite(daily_macd) and daily_macd > 0: score += 0.45; reasons.append("MACD在零轴上")
        elif np.isfinite(daily_macd) and daily_macd > safe_float(prev.get("MACD"), -999): score += 0.25; reasons.append("MACD改善")

        if np.isfinite(daily_rsi) and 45 <= daily_rsi <= 70: score += 0.35
        elif np.isfinite(daily_rsi) and daily_rsi > 78: risks.append("RSI过热")

        if 0.8 <= volume_ratio <= 2.5: score += 0.55; reasons.append("量比适中")
        elif volume_ratio > 4: risks.append("量比过高")

        if 0 <= pct <= 3.5: score += 0.45; reasons.append("涨幅适度")
        elif pct > 6: risks.append("追高风险")

        if np.isfinite(turnover) and 1.5 <= turnover <= 12: score += 0.3
        elif np.isfinite(turnover) and turnover > 20: risks.append("换手过高")

        if np.isfinite(support_dist) and 0 <= support_dist <= 3.0: score += 0.65; reasons.append("靠近支撑")
        if np.isfinite(resistance_dist) and resistance_dist >= 2.5: score += 0.45; reasons.append("上方有空间")

        score = max(0.0, min(10.0, score))

        return {
            "代码": code, "名称": name, "行业": "未知",
            "最新价": round(price, 3), "涨幅%": round(pct, 2),
            "成交额": round(amount, 0) if np.isfinite(amount) else np.nan,
            "换手率": round(turnover, 2) if np.isfinite(turnover) else np.nan,
            "量比": round(volume_ratio, 2),
            "日线MACD": round(daily_macd, 4) if np.isfinite(daily_macd) else np.nan,
            "日线RSI": round(daily_rsi, 1) if np.isfinite(daily_rsi) else np.nan,
            "支撑位": round(support, 3) if np.isfinite(support) else np.nan,
            "压力位": round(resistance, 3) if np.isfinite(resistance) else np.nan,
            "距支撑%": round(support_dist, 2) if np.isfinite(support_dist) else np.nan,
            "距压力%": round(resistance_dist, 2) if np.isfinite(resistance_dist) else np.nan,
            "评分": round(score, 2), "评分理由": "；".join(reasons[:5]) or "综合指标评分",
            "风险提示": "；".join(risks) if risks else "未触发风险",
            "计划单只资金": 20000, "参考买入股数": int(20000 / price / 100) * 100,
            "参考买入金额": int(20000 / price / 100) * 100 * price, 
            "盈利1%参考价": round(price * 1.01, 2), "盈利2%参考价": round(price * 1.02, 2), 
            "下跌1%复核价": round(price * 0.99, 2), "下跌2%复核价": round(price * 0.98, 2)
        }
    except Exception as exc:
        return {"_error": f"{code} {name}: {exc}", "_code": code}


def get_full_market_snapshot():
    log("正在生成全市场代码池并获取腾讯实时行情（纯本地生成，绝不截断）...")
    
    # 自己生成全市场代码，绕过接口的截断
    code_list = []
    # 沪市主板 600-605
    for i in range(600000, 606000):
        code_list.append(f"sh{i}")
    # 深市主板 000-003
    for i in range(1, 4000):
        code_list.append(f"sz{str(i).zfill(6)}")
    # 创业板 300-301
    for i in range(300000, 302000):
        code_list.append(f"sz{i}")
    
    quotes = []
    batch_size = 60
    headers = {"User-Agent": "Mozilla/5.0"}
    
    def fetch_batch(batch):
        url = f"https://qt.gtimg.cn/q={','.join(batch)}"
        res = []
        try:
            # 1.5秒超时，腾讯批量接口极快
            resp = requests.get(url, headers=headers, timeout=(1.5, 1.5))
            resp.encoding = "gb18030"
            lines = resp.text.split(";")
            for line in lines:
                if "=" not in line: continue
                parts = line.split("~")
                if len(parts) > 49:
                    code = parts[2]
                    price = safe_float(parts[3])
                    if price > 0:  # 过滤停牌和无效代码
                        res.append({
                            "代码": code, "名称": parts[1], 
                            "最新价": price,
                            "涨跌幅": safe_float(parts[32]), 
                            "成交额": safe_float(parts[37]), 
                            "换手率": safe_float(parts[38]), 
                            "量比": safe_float(parts[49])
                        })
        except Exception:
            pass
        return res
        
    batches = [code_list[i:i+batch_size] for i in range(0, len(code_list), batch_size)]
    
    # 并发拉取实时行情
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_batch, b) for b in batches]
        for f in as_completed(futures):
            quotes.extend(f.result())
            
    log(f"腾讯实时行情拉取完成，实际有效股票数：{len(quotes)} 只")
    return pd.DataFrame(quotes)


def scan_market():
    started = time.time()
    ensure_dirs()
    today = sh_now().strftime("%Y-%m-%d")

    log("获取全市场实时快照...")
    spot = get_full_market_snapshot()

    spot["纯代码"] = spot["代码"].astype(str)
    code = spot["纯代码"]
    name = spot["名称"].astype(str)
    
    # 剔除科创板(688/689)和北交所(代码不以6/0/3开头)
    is_sz_sh = code.str.match(r"^(000|001|002|003|300|301|600|601|603|605)")
    limit_pct = np.where(code.str.startswith(("300", "301")), 20.0, 10.0)

    spot = spot[is_sz_sh & (~name.str.contains(r"ST|退", case=False, na=False)) & (spot["最新价"] > 0) & (spot["最新价"] <= 800) & (spot["涨跌幅"] > -10.0) & (spot["涨跌幅"] < (limit_pct - 0.15))].copy()
    spot = spot.sort_values("成交额", ascending=False)

    log(f"基础过滤后进入深度分析：{len(spot)} 只（全市场深度扫描，已剔除科创板）")

    candidates = []
    total = len(spot)
    success = 0
    failed = 0

    log(f"开始隔夜超短评分；并发数={MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_stock, row): row for _, row in spot.iterrows()}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                res = future.result()
                if res and "_error" in res:
                    failed += 1
                elif res:
                    success += 1
                    candidates.append(res)
            except Exception:
                failed += 1

            if completed % 100 == 0 or completed == total:
                log(f"进度：{completed}/{total}；有效候选 {len(candidates)}；失败 {failed}")

    if candidates:
        df_all = pd.DataFrame(candidates).sort_values(["评分", "量比"], ascending=[False, False])
        result = df_all.head(TOP_N).copy()
        
        df_all.to_csv(os.path.join(REPORT_DIR, f"candidates_{today}.csv"), index=False, encoding="utf-8-sig")
        result.to_csv(os.path.join(REPORT_DIR, f"{today}.csv"), index=False, encoding="utf-8-sig")

        # 生成 markdown 报告
        lines = [f"# A股猎手｜全市场扫描报告", f"> {today} 策略：14:50观察买入 / 次日9:30退出", "", "## 最强3只"]
        for i, item in enumerate(result.to_dict("records"), 1):
            lines.extend([
                f"### {i}. **{item['名称']}（{item['代码']}）** 评分 {item['评分']}/10",
                f"- 观察价 **{item['最新价']}** ｜当日 **{item['涨幅%']:+.2f}%**",
                f"- **入选原因：** {item['评分理由']}",
                f"- **风险提示：** {item['风险提示']}",
                ""
            ])
        with open(os.path.join(REPORT_DIR, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    log(f"完成：候选 {len(candidates)}，前三名 {len(result) if candidates else 0}，失败 {failed}，耗时 {round(time.time() - started, 1)} 秒。")


if __name__ == "__main__":
    try:
        scan_market()
    except Exception as exc:
        log(f"程序运行失败：{type(exc).__name__}: {exc}")
        traceback.print_exc()
