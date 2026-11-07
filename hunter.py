# -*- coding: utf-8 -*-
"""
A股猎手升级版（纯新浪/腾讯接口版）
策略：14:50尾盘观察买入，次日9:30—9:40退出评估
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

import data_source
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TZ = ZoneInfo("Asia/Shanghai")
REPORT_DIR = "reports"
# 并发数：为了全市场扫描不被封，保持在 2。如果觉得太慢可以改成 3。
MAX_WORKERS = int(os.getenv("A_SHARE_WORKERS", "2"))

TOP_N = 3
MIN_SCORE = 0.0
HISTORY_DAYS = 120


def load_config():
    path = os.getenv("ASHARE_CONFIG", "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"配置文件读取失败：{exc}，将使用默认全市场配置")
        # 兜底默认配置，如果配置读不出来，默认扫描全市场
        return {
            "market": {
                "include_shanghai": True,
                "include_shenzhen": True,
                "include_chinext": True,
                "include_star_market": False,
                "exclude_st": True,
                "exclude_delisting": True
            },
            "screening": {
                "shortlist_size": 9999, 
                "top_n": 3
            }
        }


CONFIG = load_config()


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
    if not close_col:
        return None
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[close_col]).sort_index().reset_index(drop=True)
    if len(df) < 60:
        return None

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
    open_col = get_col(df, ["open", "开盘"])
    vol_col = get_col(df, ["volume", "成交量"])

    if high_col: df["_high"] = pd.to_numeric(df[high_col], errors="coerce")
    if low_col: df["_low"] = pd.to_numeric(df[low_col], errors="coerce")
    if open_col: df["_open"] = pd.to_numeric(df[open_col], errors="coerce")
    if vol_col: df["_volume"] = pd.to_numeric(df[vol_col], errors="coerce")

    return df


def get_headers():
    uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
    ]
    return {"User-Agent": random.choice(uas), "Referer": "https://finance.qq.com/"}


def fetch_tencent_15m(code):
    market = "sh" if str(code).startswith(("6", "9")) else "sz"
    symbol = f"{market}{code}"
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    params = {"param": f"{symbol},m15,,,160,qfq"}
    try:
        resp = requests.get(url, params=params, headers=get_headers(), timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        node = payload.get("data", {}).get(symbol, {})
        rows = node.get("m15") or node.get("qfq_m15") or []
        if not rows: return None
        df = pd.DataFrame(rows).iloc[:, :6]
        df.columns = ["时间", "开盘", "收盘", "最高", "最低", "成交量"]
        for col in ["开盘", "收盘", "最高", "最低", "成交量"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["收盘"]).reset_index(drop=True)
    except Exception:
        return None


def fetch_sina_15m(code):
    market = "sh" if str(code).startswith(("6", "9")) else "sz"
    symbol = f"{market}{code}"
    url = f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_{symbol}_15m=/CN_MarketDataService.getKLineData"
    params = {"symbol": symbol, "scale": "15", "ma": "no", "datalen": "160"}
    headers = {"User-Agent": random.choice(["Mozilla/5.0", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"]), "Referer": "https://finance.sina.com.cn/"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        text = resp.text
        match = re.search(r"=\s*\((\[.*\])\);", text, re.S)
        if not match: return None
        rows = json.loads(match.group(1))
        if not rows: return None
        df = pd.DataFrame(rows)
        df = df.rename(columns={"day": "时间", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"})
        for col in ["开盘", "收盘", "最高", "最低", "成交量"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["收盘"]).reset_index(drop=True)
    except Exception:
        return None


def fetch_intraday_15m(code):
    time.sleep(random.uniform(0.1, 0.4))
    df = fetch_tencent_15m(code)
    if df is None or len(df) < 30:
        df = fetch_sina_15m(code)
    if df is None or len(df) < 30:
        return None

    close = df["收盘"]
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    df["DIF"] = dif
    df["DEA"] = dea
    df["MACD"] = 2 * (dif - dea)
    df["RSI15"] = calculate_rsi(close, 14)
    return df


def market_temperature(snapshot):
    df = snapshot.copy()
    pct = pd.to_numeric(df.get("涨跌幅"), errors="coerce")
    amount = pd.to_numeric(df.get("成交额"), errors="coerce")
    valid = pct.dropna()
    if valid.empty:
        return {"上涨家数": 0, "下跌家数": 0, "平盘家数": 0, "上涨比例": 0, "跌停附近": 0, "涨停附近": 0, "成交额亿元": None, "热度": "数据不足"}

    up = int((valid > 0).sum())
    down = int((valid < 0).sum())
    flat = int((valid == 0).sum())
    up_limit = int((valid >= 9.5).sum())
    down_limit = int((valid <= -9.5).sum())
    breadth = up / max(1, len(valid))
    total_amount = float(amount.fillna(0).sum() / 1e8)

    if breadth >= .65 and up_limit >= 50 and down_limit <= 10: mood = "偏热"
    elif breadth >= .52 and down > 0 and down_limit <= 20: mood = "温和偏强"
    elif breadth <= .35 or down_limit >= 30: mood = "偏冷"
    else: mood = "分化"

    return {"上涨家数": up, "下跌家数": down, "平盘家数": flat, "上涨比例": round(breadth * 100, 1), "涨停附近": up_limit, "跌停附近": down_limit, "成交额亿元": round(total_amount, 1), "热度": mood}


def build_sector_heat(snapshot):
    if "行业" not in snapshot.columns: return pd.DataFrame(), "行情快照未提供行业字段，板块热度不可计算"
    df = snapshot.copy()
    df["涨跌幅"] = pd.to_numeric(df.get("涨跌幅"), errors="coerce")
    df["成交额"] = pd.to_numeric(df.get("成交额"), errors="coerce")
    df = df.dropna(subset=["行业", "涨跌幅"])
    if df.empty: return pd.DataFrame(), "行业字段无有效数据"

    grouped = df.groupby("行业").agg(
        股票数=("代码", "count"),
        平均涨幅=("涨跌幅", "mean"),
        上涨家数=("涨跌幅", lambda x: int((x > 0).sum())),
        成交额亿元=("成交额", lambda x: float(x.fillna(0).sum() / 1e8))
    ).reset_index()
    grouped["上涨比例"] = (grouped["上涨家数"] / grouped["股票数"] * 100)
    grouped = grouped.sort_values(["平均涨幅", "上涨比例", "成交额亿元"], ascending=False).head(15)
    grouped["资金净流入"] = "当前数据源未提供，不能确认"
    return grouped, "板块热度为行业涨幅/广度/成交额代理，不等于主力资金净流入"


def load_previous_picks(today):
    ensure_dirs()
    candidates = []
    run_dir = os.path.join(REPORT_DIR, "runs")
    if os.path.isdir(run_dir):
        for filename in os.listdir(run_dir):
            match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_(\d{6})\.csv", filename)
            if match and match.group(1) < today:
                candidates.append((match.group(1) + " " + match.group(2), os.path.join(run_dir, filename)))
    if candidates:
        stamp, path = sorted(candidates, reverse=True)[0]
        try:
            df = pd.read_csv(path, dtype={"代码": str})
            if not df.empty and "代码" in df.columns: df["代码"] = df["代码"].map(normalize_code)
            return stamp, df
        except Exception: pass

    daily = []
    for filename in os.listdir(REPORT_DIR):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.csv", filename):
            date_text = filename[:-4]
            if date_text < today: daily.append((date_text, os.path.join(REPORT_DIR, filename)))
    if not daily: return None, pd.DataFrame()

    prev_date, path = sorted(daily, reverse=True)[0]
    try:
        df = pd.read_csv(path, dtype={"代码": str})
        if not df.empty and "代码" in df.columns: df["代码"] = df["代码"].map(normalize_code)
        return prev_date, df
    except Exception:
        return prev_date, pd.DataFrame()


def learning_paths():
    return {
        "outcomes": os.path.join(REPORT_DIR, "learning_outcomes.csv"),
        "memory": os.path.join(REPORT_DIR, "learning_memory.json"),
        "rules": os.path.join(REPORT_DIR, "learned_rules.json"),
    }


def read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: pass
    return default


def diagnose_drop(old, current_price, current_pct, market_pct=None):
    reasons = []
    old_price = safe_float(old.get("最新价"))
    old_rsi = safe_float(old.get("RSI"))
    old_dev = safe_float(old.get("偏离MA10%"))
    old_turnover = safe_float(old.get("换手率"))
    old_volume = safe_float(old.get("量比"))
    old_macd = safe_float(old.get("MACD"))
    old_pct = safe_float(old.get("涨幅%", old.get("涨跌幅")))
    change = ((current_price / old_price - 1) * 100 if old_price > 0 and current_price > 0 else np.nan)

    if np.isfinite(change) and change <= -2:
        if market_pct is not None and pd.notna(market_pct) and market_pct <= -1: reasons.append("市场整体走弱")
        if np.isfinite(old_rsi) and old_rsi >= 70: reasons.append("入选时RSI偏高")
        if np.isfinite(old_dev) and old_dev >= 5: reasons.append("入选时明显偏离MA10")
        if np.isfinite(old_turnover) and old_turnover >= 15: reasons.append("入选时换手率偏高")
        if np.isfinite(old_volume) and old_volume >= 3: reasons.append("入选时量比偏高")
        if np.isfinite(old_macd) and old_macd < 0: reasons.append("入选时MACD仍在零轴下方")
        if np.isfinite(old_pct) and old_pct >= 5: reasons.append("入选当日涨幅较大")
        if not reasons: reasons.append("已确认价格走弱，需进一步核对公告")
    elif np.isfinite(change) and change < 0: reasons.append("小幅回撤")
    elif np.isfinite(change): reasons.append("未出现下跌型失误")
    else: reasons.append("价格数据不完整")
    return change, "；".join(reasons)


def update_learning(previous_date, previous_df, current_spot, today):
    paths = learning_paths()
    outcomes_path = paths["outcomes"]
    if previous_df is None or previous_df.empty: return pd.DataFrame(), ["没有上一交易日推荐记录"]

    snap = current_spot.copy()
    snap["纯代码"] = snap["代码"].map(normalize_code)
    snap = snap.drop_duplicates("纯代码").set_index("纯代码")
    outcome_rows = []

    for _, old in previous_df.iterrows():
        code = normalize_code(old.get("代码", ""))
        if not code or code not in snap.index: continue
        current = snap.loc[code]
        current_price = safe_float(current.get("最新价"))
        current_pct = safe_float(current.get("涨跌幅"))
        old_price = safe_float(old.get("最新价"))
        change, diagnosis = diagnose_drop(old, current_price, current_pct)
        if not np.isfinite(change): continue

        outcome_rows.append({
            "推荐日期": str(previous_date).split(" ")[0], "复盘日期": today,
            "代码": code, "名称": str(old.get("名称", "")), "入选评分": safe_float(old.get("评分")),
            "入选价": old_price, "复盘价": current_price, "T1收益%": round(change, 3),
            "是否下跌": int(change < 0), "是否明显下跌": int(change <= -2),
            "RSI区间": ("偏热" if safe_float(old.get("RSI")) >= 70 else "非偏热"),
            "乖离区间": ("高乖离" if safe_float(old.get("偏离MA10%")) >= 5 else "普通乖离"),
            "量比区间": ("高量比" if safe_float(old.get("量比")) >= 3 else "普通量比"),
            "换手区间": ("高换手" if safe_float(old.get("换手率")) >= 15 else "普通换手"),
            "MACD区间": ("零轴下" if safe_float(old.get("MACD")) < 0 else "零轴上"),
            "下跌归因候选": diagnosis,
        })

    if not outcome_rows: return pd.DataFrame(), ["未匹配到有效价格"]
    new_df = pd.DataFrame(outcome_rows)

    if os.path.exists(outcomes_path):
        try:
            old_df = pd.read_csv(outcomes_path, dtype={"代码": str})
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception: pass

    new_df["代码"] = new_df["代码"].map(normalize_code)
    new_df = new_df.drop_duplicates(["推荐日期", "复盘日期", "代码"], keep="last")
    new_df.to_csv(outcomes_path, index=False, encoding="utf-8-sig")

    feature_cols = ["RSI区间", "乖离区间", "量比区间", "换手区间", "MACD区间"]
    rules = {"说明": "根据历史推荐T+1结果统计", "规则": {}}

    for col in feature_cols:
        for bucket, group in new_df.groupby(col, dropna=False):
            if len(group) < 8: continue
            loss_rate = float((pd.to_numeric(group["T1收益%"], errors="coerce") <= -2).mean())
            avg_return = float(pd.to_numeric(group["T1收益%"], errors="coerce").mean())
            penalty = min(1.0, max(0.0, (loss_rate - 0.40) * 2.0)) if avg_return < 0 else 0.0
            rules["规则"][f"{col}={bucket}"] = {"样本数": int(len(group)), "明显下跌率": round(loss_rate, 3), "平均T1收益%": round(avg_return, 3), "建议扣分": round(penalty, 2)}

    with open(paths["rules"], "w", encoding="utf-8") as f: json.dump(rules, f, ensure_ascii=False, indent=2)

    memory = read_json(paths["memory"], {"累计复盘次数": 0, "累计样本数": 0, "最近复盘日期": None, "经验摘要": []})
    memory["累计复盘次数"] = int(memory.get("累计复盘次数", 0)) + 1
    memory["累计样本数"] = int(len(new_df))
    memory["最近复盘日期"] = today
    drops = new_df[pd.to_numeric(new_df["T1收益%"], errors="coerce") <= -2]
    summary = f"截至{today}累计{len(new_df)}条T+1样本；明显下跌样本{len(drops)}条。"
    memory.setdefault("经验摘要", []).append(summary)
    memory["经验摘要"] = memory["经验摘要"][-30:]
    with open(paths["memory"], "w", encoding="utf-8") as f: json.dump(memory, f, ensure_ascii=False, indent=2)

    return new_df, [f"已更新学习样本：本轮新增{len(outcome_rows)}条，累计{len(new_df)}条。"]


def apply_learned_penalties(candidates):
    paths = learning_paths()
    rules = read_json(paths["rules"], {}).get("规则", {})
    if not rules or not candidates: return candidates

    for item in candidates:
        features = {
            "RSI区间": ("偏热" if safe_float(item.get("RSI")) >= 70 else "非偏热"),
            "乖离区间": ("高乖离" if safe_float(item.get("偏离MA10%")) >= 5 else "普通乖离"),
            "量比区间": ("高量比" if safe_float(item.get("量比")) >= 3 else "普通量比"),
            "换手区间": ("高换手" if safe_float(item.get("换手率")) >= 15 else "普通换手"),
            "MACD区间": ("零轴下" if safe_float(item.get("MACD")) < 0 else "零轴上"),
        }
        penalty = 0.0
        warnings_found = []
        for col, bucket in features.items():
            rule = rules.get(f"{col}={bucket}", {})
            p = safe_float(rule.get("建议扣分"), 0.0)
            if p > 0:
                penalty += p
                warnings_found.append(f"历史学习提示：{col}={bucket}")
        penalty = min(1.5, penalty)
        item["学习扣分"] = round(penalty, 2)
        if penalty:
            item["评分"] = round(max(0.0, safe_float(item.get("评分"), 0.0) - penalty), 2)
            existing = str(item.get("风险提示", ""))
            item["风险提示"] = "；".join(x for x in [existing, *warnings_found] if x and x != "未触发主要风险项")
    return candidates


def fetch_hist(code, today_compact):
    try:
        time.sleep(random.uniform(0.1, 0.4))
        hist = data_source.get_history(code, CONFIG)
        if hist is not None and len(hist) >= 60: return hist
        raise RuntimeError("历史行情为空或不足60条")
    except Exception as exc:
        raise RuntimeError(f"历史行情获取失败：{exc}")


def process_single_stock(row, today_compact, board_map=None):
    raw_code = str(row.get("代码", ""))
    code = normalize_code(raw_code)
    name = str(row.get("名称", ""))
    if not code or "ST" in name.upper() or "退" in name: return None

    try:
        hist_raw = fetch_hist(code, today_compact)
        hist = calculate_indicators(hist_raw)
        if hist is None or len(hist) < 60: raise RuntimeError("日线历史不足60根")

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
        high60 = safe_float(prev_hist[high_col].max()) if high_col else np.nan
        low60 = safe_float(prev_hist[low_col].min()) if low_col else np.nan

        support = max([x for x in [low20, ma20, ma60] if np.isfinite(x) and x < price], default=np.nan)
        resistance = min([x for x in [high20, high60] if np.isfinite(x) and x > price], default=np.nan)
        support_dist = ((price / support - 1) * 100 if np.isfinite(support) and support > 0 else np.nan)
        resistance_dist = ((resistance / price - 1) * 100 if (np.isfinite(resistance) and price > 0) else np.nan)

        # 如果拿不到15分钟数据，这里会返回 None，不会抛出异常导致整个股票跳过
        intraday = fetch_intraday_15m(code)
        m15_macd, m15_prev, m15_dif, m15_dea, m15_rsi = np.nan, np.nan, np.nan, np.nan, np.nan
        m15_volume_trend = "分钟数据不可用"

        # 只有拿到了 15 分钟数据才计算 15 分钟指标
        if intraday is not None and len(intraday) >= 3:
            ilast = intraday.iloc[-1]
            iprev = intraday.iloc[-2]
            m15_macd = safe_float(ilast.get("MACD"))
            m15_prev = safe_float(iprev.get("MACD"))
            m15_dif = safe_float(ilast.get("DIF"))
            m15_dea = safe_float(ilast.get("DEA"))
            m15_rsi = safe_float(ilast.get("RSI15"))
            v = pd.to_numeric(intraday["成交量"].tail(6), errors="coerce")
            if len(v) >= 6 and v.iloc[-3:].mean() > v.iloc[:3].mean(): m15_volume_trend = "尾盘量能增强"
            elif len(v) >= 6: m15_volume_trend = "尾盘量能平稳/减弱"

        score = 0.0
        reasons, risks = [], []

        # ===== 15分钟指标评分（如果不可用会跳过，不影响下面日线指标评分） =====
        if np.isfinite(m15_macd) and np.isfinite(m15_prev):
            if m15_macd > m15_prev: score += 1.1; reasons.append("15分钟MACD改善")
            else: risks.append("15分钟MACD走弱")
            if np.isfinite(m15_dif) and np.isfinite(m15_dea) and m15_dif > m15_dea: score += 0.7; reasons.append("15分钟MACD处于多头")
            else: risks.append("15分钟MACD未形成多头")
        else: risks.append("缺少15分钟数据，尾盘信号不完整")

        if np.isfinite(m15_rsi):
            if 48 <= m15_rsi <= 68: score += 0.8; reasons.append("15分钟RSI强度适中")
            elif 42 <= m15_rsi < 48: score += 0.35
            elif m15_rsi >= 78: risks.append("15分钟RSI过热")
        else: risks.append("15分钟RSI不可用")

        # ===== 日线指标评分（无论 15 分钟拉没拉到，都会继续执行） =====
        if price >= ma5 and price >= ma10: score += 0.8; reasons.append("价格站在MA5/MA10上方")
        elif price >= ma20: score += 0.35
        else: risks.append("价格低于MA20")

        if np.isfinite(ma20) and np.isfinite(ma60) and ma20 >= ma60: score += 0.6; reasons.append("MA20在MA60上方")
        else: risks.append("日线中期趋势偏弱")

        if np.isfinite(daily_macd) and daily_macd > 0: score += 0.45; reasons.append("日线MACD在零轴上方")
        elif np.isfinite(daily_macd) and daily_macd > safe_float(prev.get("MACD"), -999): score += 0.25; reasons.append("日线MACD边际改善")

        if np.isfinite(daily_rsi) and 45 <= daily_rsi <= 70: score += 0.35
        elif np.isfinite(daily_rsi) and daily_rsi > 78: risks.append("日线RSI过热")

        if 0.8 <= volume_ratio <= 2.5: score += 0.55; reasons.append("日内量比适中")
        elif volume_ratio > 4: risks.append("量比过高，警惕尾盘分歧")

        if 0 <= pct <= 3.5: score += 0.45; reasons.append("当日涨幅不过度")
        elif pct > 6: risks.append("当日涨幅过大，隔夜追高风险")

        if np.isfinite(turnover) and 1.5 <= turnover <= 12: score += 0.3
        elif np.isfinite(turnover) and turnover > 20: risks.append("换手率异常偏高")

        if np.isfinite(support_dist):
            if 0 <= support_dist <= 3.0: score += 0.65; reasons.append("靠近20/60日支撑区")
            elif support_dist > 8: risks.append("距离下方支撑较远")

        if np.isfinite(resistance_dist):
            if resistance_dist <= 1.2: risks.append("临近20/60日压力位，突破空间有限")
            elif resistance_dist >= 2.5: score += 0.45; reasons.append("上方短期压力空间尚可")

        if np.isfinite(support_dist) and support_dist <= 2.0 and m15_volume_trend == "尾盘量能增强": score += 0.35; reasons.append("支撑附近出现量能增强")
        if m15_volume_trend == "尾盘量能增强": score += 0.25

        limit_pct = (20.0 if code.startswith(("300", "301")) else 10.0)
        if pct <= -10.0 or pct >= limit_pct - 0.15 or price > 800: return None

        for risk, penalty in [
            ("15分钟MACD走弱", 0.6), ("15分钟MACD未形成多头", 0.4), ("15分钟RSI过热", 0.5),
            ("价格低于MA20", 0.6), ("日线中期趋势偏弱", 0.4), ("当日涨幅过大，隔夜追高风险", 0.8),
            ("量比过高，警惕尾盘分歧", 0.5), ("换手率异常偏高", 0.4), ("临近20/60日压力位，突破空间有限", 0.8),
            ("距离下方支撑较远", 0.3), ("缺少15分钟数据，尾盘信号不完整", 0.8)
        ]:
            if risk in risks: score -= penalty

        score = max(0.0, min(10.0, score))
        overnight_cfg = CONFIG.get("overnight", {})
        capital_per_stock = float(overnight_cfg.get("capital_per_stock", 20000))
        planned_shares = int(capital_per_stock / price / 100) * 100
        planned_amount = round(planned_shares * price, 2)

        return {
            "代码": code, "名称": name, "行业": str(row.get("行业", "未知")),
            "最新价": round(price, 3), "涨幅%": round(pct, 2),
            "成交额": round(amount, 0) if np.isfinite(amount) else np.nan,
            "换手率": round(turnover, 2) if np.isfinite(turnover) else np.nan,
            "量比": round(volume_ratio, 2),
            "日线MACD": round(daily_macd, 4) if np.isfinite(daily_macd) else np.nan,
            "日线RSI": round(daily_rsi, 1) if np.isfinite(daily_rsi) else np.nan,
            "15分钟MACD": round(m15_macd, 4) if np.isfinite(m15_macd) else np.nan,
            "15分钟RSI": round(m15_rsi, 1) if np.isfinite(m15_rsi) else np.nan,
            "尾盘量能": m15_volume_trend,
            "支撑位": round(support, 3) if np.isfinite(support) else np.nan,
            "压力位": round(resistance, 3) if np.isfinite(resistance) else np.nan,
            "距支撑%": round(support_dist, 2) if np.isfinite(support_dist) else np.nan,
            "距压力%": round(resistance_dist, 2) if np.isfinite(resistance_dist) else np.nan,
            "评分": round(score, 2), "评分理由": "；".join(reasons[:5]) or "综合指标评分",
            "风险提示": "；".join(risks) if risks else "未触发主要风险项",
            "计划单只资金": capital_per_stock, "参考买入股数": planned_shares,
            "参考买入金额": planned_amount, "盈利1%参考价": round(price * 1.01, 2),
            "盈利2%参考价": round(price * 1.02, 2), "下跌1%复核价": round(price * 0.99, 2),
            "下跌2%复核价": round(price * 0.98, 2),
            "历史K线日期": str(last.get(get_col(hist, ["date", "日期"], hist.columns[0])))[:10]
        }

    except Exception as exc:
        return {"_error": f"{code} {name}: {type(exc).__name__}: {exc}", "_code": code}


def build_fallback_candidates(spot, existing, top_n=3):
    existing_codes = {normalize_code(x.get("代码")) for x in existing}
    fallback_rows = []

    for _, row in spot.iterrows():
        code = normalize_code(row.get("代码"))
        if not code or code in existing_codes: continue
        price, pct = safe_float(row.get("最新价")), safe_float(row.get("涨跌幅"))
        if not np.isfinite(price) or price <= 0 or price > 800 or not np.isfinite(pct): continue
        if pct <= -10: continue
        
        limit_pct = (20.0 if code.startswith(("300", "301")) else 10.0)
        if pct >= limit_pct - 0.15: continue

        volume_ratio = safe_float(row.get("量比"), 0.0)
        turnover = safe_float(row.get("换手率"), np.nan)
        score = 0.5
        reasons = ["实时快照可用，但历史/分钟指标未能完整取得"]
        risks = ["技术指标数据不完整，排名仅作备选观察，不宜直接下单"]

        if 0.3 <= pct <= 3.0: score += 0.8; reasons.append("当日涨幅温和")
        elif pct > 5: risks.append("当日涨幅偏大，隔夜追高风险"); score -= 0.4
        if 0.8 <= volume_ratio <= 2.5: score += 0.5; reasons.append("量比处于活跃区间")
        elif volume_ratio > 4: score -= 0.3; risks.append("量比过高")
        if np.isfinite(turnover) and 1 <= turnover <= 12: score += 0.3

        entry = round(price, 3)
        capital_per_stock = float(CONFIG.get("overnight", {}).get("capital_per_stock", 20000))
        planned_shares = int(capital_per_stock / price / 100) * 100

        fallback_rows.append({
            "代码": code, "名称": str(row.get("名称", "")), "行业": str(row.get("行业", "未知")),
            "最新价": entry, "涨幅%": round(pct, 2), "成交额": safe_float(row.get("成交额")),
            "换手率": turnover, "量比": volume_ratio, "日线MACD": np.nan, "日线RSI": np.nan,
            "15分钟MACD": np.nan, "15分钟RSI": np.nan, "尾盘量能": "分钟数据不可用",
            "支撑位": np.nan, "压力位": np.nan, "距支撑%": np.nan, "距压力%": np.nan,
            "评分": round(max(0, min(3, score)), 2), "评分理由": "；".join(reasons),
            "风险提示": "；".join(risks), "计划单只资金": capital_per_stock,
            "参考买入股数": planned_shares, "参考买入金额": round(planned_shares * entry, 2),
            "盈利1%参考价": round(entry * 1.01, 2), "盈利2%参考价": round(entry * 1.02, 2),
            "下跌1%复核价": round(entry * 0.99, 2), "下跌2%复核价": round(entry * 0.98, 2),
            "数据完整性": "快照备选-技术数据缺失"
        })

    fallback_rows.sort(key=lambda x: (x["评分"], safe_float(x.get("成交额"), 0)), reverse=True)
    return fallback_rows[:max(0, top_n - len(existing))]


def get_spot_snapshot():
    log("使用新浪/腾讯接口获取全市场快照...")
    data_source.reset_data_source_status()
    snapshot = data_source.get_market_snapshot(CONFIG)
    if not snapshot: raise RuntimeError("实时快照接口返回空数据")
    spot = pd.DataFrame(snapshot)
    spot = spot.rename(columns={"code": "代码", "name": "名称", "price": "最新价", "change_pct": "涨跌幅", "amount": "成交额", "turnover_rate": "换手率"})
    for col in ["量比", "委比", "外盘", "内盘"]:
        if col not in spot.columns: spot[col] = np.nan
    for col in ["最新价", "涨跌幅", "成交额", "量比", "委比", "外盘", "内盘", "换手率"]:
        if col in spot.columns: spot[col] = pd.to_numeric(spot[col], errors="coerce")
    return spot, "sina_tencent"


def review_previous_picks(previous_date, previous_df, current_spot):
    if previous_df is None or previous_df.empty: return pd.DataFrame(), ["没有找到更早日期的推荐 CSV"]
    current = current_spot.copy()
    current["代码"] = current["代码"].map(normalize_code)
    current = current.drop_duplicates("代码").set_index("代码")
    records = [f"上次推荐日期：{previous_date}"]

    for _, old in previous_df.iterrows():
        code = normalize_code(old.get("代码", ""))
        name, old_price, old_score = str(old.get("名称", "")), safe_float(old.get("最新价")), safe_float(old.get("评分"))
        if code in current.index:
            new_price = safe_float(current.loc[code].get("最新价"))
            pct_now = safe_float(current.loc[code].get("涨跌幅"))
            change = ((new_price / old_price - 1) * 100 if (old_price > 0 and new_price > 0) else np.nan)
            judgment = "价格数据不完整"
            if np.isfinite(change):
                if change >= 2: judgment = "价格表现偏强"
                elif change > -1: judgment = "价格暂时横盘/小幅波动"
                else: judgment = "价格走弱，重新评估"
            _, diagnosis = diagnose_drop(old, new_price, pct_now)
            records.append({"上次推荐日期": previous_date, "代码": code, "名称": name, "上次评分": old_score, "上次参考价": old_price, "当前价": new_price, "相对上次价格变化%": round(change, 2) if np.isfinite(change) else np.nan, "当前日涨幅%": pct_now, "复盘判断": judgment, "下跌归因候选": diagnosis})
        else:
            records.append({"上次推荐日期": previous_date, "代码": code, "名称": name, "上次评分": old_score, "上次参考价": old_price, "当前价": np.nan, "相对上次价格变化%": np.nan, "当前日涨幅%": np.nan, "复盘判断": "当前快照未找到该股票"})
    return pd.DataFrame(records), ["复盘完成"]


def write_report(today, candidates_df, result, previous_date, review_df, board_df, statuses, scan_stats, market_stats, sector_df, sector_status):
    ensure_dirs()
    lines = [
        "# A股猎手｜隔夜超短报告",
        f"> {today}  ·  扫描时间 {sh_now().strftime('%H:%M:%S')}  ·  策略：14:50尾盘观察买入 / 次日9:30—9:40退出评估",
        "", "## 市场温度",
        f"**{market_stats.get('热度','未知')}**　上涨 {market_stats.get('上涨家数',0)} 家 / 下跌 {market_stats.get('下跌家数',0)} 家　｜上涨比例 {market_stats.get('上涨比例',0)}%",
        f"涨停附近 {market_stats.get('涨停附近',0)}　·　跌停附近 {market_stats.get('跌停附近',0)}　·　两市成交额约 {market_stats.get('成交额亿元','未知')} 亿元",
        "", "## 今日板块热度 TOP",
        f"> {sector_status}；不能把板块成交额/涨幅代理误称为主力净流入。",
    ]

    if sector_df is not None and not sector_df.empty:
        lines.extend(["|行业|平均涨幅|上涨比例|成交额(亿元)|资金净流入|", "|---|---:|---:|---:|---|"])
        for _, r in sector_df.head(8).iterrows():
            lines.append(f"|{r['行业']}|{r['平均涨幅']:.2f}%|{r['上涨比例']:.1f}%|{r['成交额亿元']:.1f}|未取得可靠数据|")
    else:
        lines.append("- 当前快照未提供可用行业分类，板块热度无法计算。")

    lines.extend(["", "## 最强3只｜隔夜观察", "前两名为买入候选，每只计划资金20,000元；第三名备用观察。", ""])
    if result is None or result.empty:
        lines.append("符合硬性条件的实时行情不足3只，或行情快照不可用；不生成虚假股票。")
    else:
        for i, item in enumerate(result.to_dict("records"), 1):
            lines.extend([
                f"### {i}. **{item['名称']}（{item['代码']}）**　<small>评分 {item['评分']}/10</small>",
                f"- 观察价 **{item['最新价']}**　｜当日 **{item['涨幅%']:+.2f}%**　｜行业 **{item.get('行业','未知')}**",
                f"- **入选原因：** {item.get('评分理由','暂无')}",
                f"- **支撑/压力：** {item.get('支撑位','—')} / {item.get('压力位','—')}　｜距支撑 {item.get('距支撑%','—')}%　｜距压力 {item.get('距压力%','—')}%",
                f"- **尾盘信号：** 15分钟MACD {item.get('15分钟MACD','—')}；15分钟RSI {item.get('15分钟RSI','—')}；{item.get('尾盘量能','—')}",
                f"- **买入安排：** {'前两名买入候选' if i <= 2 else '第三名备用观察'}；单只计划资金 {item.get('计划单只资金', 20000):.0f} 元；按参考价估算 {item.get('参考买入股数', 0)} 股（100股整数倍）",
                f"- **止盈计划：** +1%参考价 {item.get('盈利1%参考价','—')} 元，卖出持仓50%；+2%参考价 {item.get('盈利2%参考价','—')} 元，卖出剩余全部",
                f"- **风险复核：** -1%参考价 {item.get('下跌1%复核价','—')} 元触发第一次复核；-2%参考价 {item.get('下跌2%复核价','—')} 元触发第二次复核，均不自动强制清仓",
                f"- **风险：** {item.get('风险提示','无')}", ""
            ])

    lines.extend(["## 候选名单", ""])
    if candidates_df is not None and not candidates_df.empty:
        lines.extend(["|排名|代码|名称|评分|当日涨幅|15分钟MACD|15分钟RSI|支撑|压力|行业|", "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"])
        for i, r in enumerate(candidates_df.head(50).to_dict("records"), 1):
            lines.append(f"|{i}|{r.get('代码')}|{r.get('名称')}|{r.get('评分')}|{r.get('涨幅%')}%|{r.get('15分钟MACD')}|{r.get('15分钟RSI')}|{r.get('支撑位')}|{r.get('压力位')}|{r.get('行业','未知')}|")
    else:
        lines.append("暂无候选。")

    lines.extend(["", "## 上一轮复盘", ""])
    if previous_date and review_df is not None and not review_df.empty:
        lines.append(f"上次记录：{previous_date}")
        for _, r in review_df.iterrows():
            chg = r.get("相对上次价格变化%")
            chg_text = f"{chg:+.2f}%" if pd.notna(chg) else "无法计算"
            lines.append(f"- {r.get('名称')}（{r.get('代码')}）：{chg_text}；{r.get('复盘判断')}；{r.get('下跌归因候选','需进一步核实')}")
    else:
        lines.append("暂无可复盘的上一轮记录。")

    lines.extend(["", "## 数据与执行提醒",
        f"- 历史日线成功 {scan_stats.get('success',0)} 只，失败 {scan_stats.get('failed',0)} 只。",
        "- 本程序生成选股和交易计划，不连接券商自动下单。",
        "- A股T+1：当日买入不能当日卖出；跳空、跌停、流动性不足可能无法按预期成交。"])

    with open(os.path.join(REPORT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def scan_market():
    started = time.time()
    ensure_dirs()
    now = sh_now()
    today = now.strftime("%Y-%m-%d")
    today_compact = now.strftime("%Y%m%d")

    log("获取全市场实时快照...")
    spot, snapshot_source = get_spot_snapshot()
    spot_all = spot.copy()

    spot["纯代码"] = spot["代码"].map(normalize_code)
    for col in ["最新价", "涨跌幅", "成交额", "量比", "委比", "外盘", "内盘", "换手率"]:
        if col not in spot.columns: spot[col] = np.nan
        spot[col] = pd.to_numeric(spot[col], errors="coerce")

    code = spot["纯代码"].astype(str)
    name = spot["名称"].astype(str)

    is_sz_sh = code.str.match(r"^(000|001|002|003|300|301|600|601|603|605)")
    limit_pct = np.where(code.str.startswith(("300", "301")), 20.0, 10.0)

    spot = spot[
        is_sz_sh
        & (~name.str.contains(r"ST|退", case=False, na=False))
        & (spot["最新价"] > 0)
        & (spot["最新价"] <= 800)
        & (spot["涨跌幅"] > -10.0)
        & (spot["涨跌幅"] < (limit_pct - 0.15))
    ].copy()

    spot["_amount_sort"] = spot["成交额"].fillna(0)
    # 不再强制写死 800，根据配置读取，如果配置没读出来默认 9999 全市场
    max_candidates = int(CONFIG.get("screening", {}).get("shortlist_size", 9999))
    spot = spot.sort_values("_amount_sort", ascending=False).head(max_candidates).drop(columns=["_amount_sort"])

    log(f"基础过滤后进入深度分析：{len(spot)} 只（全市场扫描）")

    market_stats = market_temperature(spot_all)
    sector_df, sector_status = build_sector_heat(spot_all)
    previous_date, previous_df = load_previous_picks(today)
    review_df, review_notes = review_previous_picks(previous_date, previous_df, spot_all)
    update_learning(previous_date, previous_df, spot_all, today)

    candidates = []
    errors = []
    total = len(spot)
    success = 0
    failed = 0

    log(f"开始隔夜超短评分；并发数={MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_stock, row, today_compact, None): row for _, row in spot.iterrows()}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            try:
                res = future.result()
                if res and "_error" in res:
                    failed += 1
                    errors.append({"时间": sh_now().strftime("%Y-%m-%d %H:%M:%S"), "错误": res["_error"]})
                elif res:
                    success += 1
                    candidates.append(res)
            except Exception as exc:
                failed += 1
                row = futures[future]
                errors.append({"时间": sh_now().strftime("%Y-%m-%d %H:%M:%S"), "错误": f"{row.get('代码')} {row.get('名称')}: {exc}"})

            if completed % 100 == 0 or completed == total:
                log(f"进度：{completed}/{total}；有效候选 {len(candidates)}；失败 {failed}")

    error_path = os.path.join(REPORT_DIR, "scan_errors.csv")
    if errors:
        pd.DataFrame(errors).to_csv(error_path, mode="a", header=not os.path.exists(error_path), index=False, encoding="utf-8-sig")

    top_n = int(CONFIG.get("screening", {}).get("top_n", 3))
    if candidates: candidates = apply_learned_penalties(candidates)

    # 把全市场抓日线失败的股票，降级扔进实时备选池
    fallback_candidates = build_fallback_candidates(spot, candidates, top_n=top_n)
    candidates.extend(fallback_candidates)

    if candidates:
        df_all = pd.DataFrame(candidates).sort_values(["评分", "量比"], ascending=[False, False])
        result = df_all.head(top_n).copy()
        df_all.to_csv(os.path.join(REPORT_DIR, f"candidates_{today}.csv"), index=False, encoding="utf-8-sig")
        result.to_csv(os.path.join(REPORT_DIR, f"{today}.csv"), index=False, encoding="utf-8-sig")

        history_path = os.path.join(REPORT_DIR, "recommendation_history.csv")
        history = result.copy()
        history.insert(0, "推荐日期", today)
        if os.path.exists(history_path):
            try:
                old_hist = pd.read_csv(history_path, dtype={"代码": str})
                history = pd.concat([old_hist, history], ignore_index=True)
                history["代码"] = history["代码"].map(normalize_code)
                history = history.drop_duplicates(["推荐日期", "代码"], keep="last")
            except Exception: pass
        history.to_csv(history_path, index=False, encoding="utf-8-sig")
    else:
        df_all, result = pd.DataFrame(), pd.DataFrame()
        pd.DataFrame(columns=["代码", "名称", "评分"]).to_csv(os.path.join(REPORT_DIR, f"candidates_{today}.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["代码", "名称", "评分"]).to_csv(os.path.join(REPORT_DIR, f"{today}.csv"), index=False, encoding="utf-8-sig")

    if sector_df is not None and not sector_df.empty:
        sector_df.to_csv(os.path.join(REPORT_DIR, f"sector_heat_{today}.csv"), index=False, encoding="utf-8-sig")

    stats = {"snapshot_source": snapshot_source, "universe": total, "success": success, "failed": failed, "seconds": round(time.time() - started, 1)}
    write_report(today, df_all, result, previous_date, review_df, pd.DataFrame(), {"行业": sector_status, "资金": "净流入数据不可用"}, stats, market_stats, sector_df, sector_status)
    log(f"完成：候选 {len(df_all)}，前三名 {len(result)}，失败 {failed}，耗时 {stats['seconds']} 秒。")


if __name__ == "__main__":
    try:
        scan_market()
    except Exception as exc:
        ensure_dirs()
        log(f"程序运行失败：{type(exc).__name__}: {exc}")
        traceback.print_exc()
        with open(os.path.join(REPORT_DIR, "fatal_error.log"), "a", encoding="utf-8") as f:
            f.write(f"\n[{sh_now().isoformat()}] {type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc())
