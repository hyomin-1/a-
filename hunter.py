# -*- coding: utf-8 -*-
"""A股猎手 V4：双路线100分制尾盘隔夜扫描与复盘。"""
import os
import json
import time
import traceback
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from data_source import (
    get_market_snapshot, get_history,
    get_data_source_status, reset_data_source_status
)

TZ = ZoneInfo("Asia/Shanghai")
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def now_cn():
    return datetime.now(TZ)


def log(msg):
    print(f"[{now_cn():%H:%M:%S}] {msg}", flush=True)


def fnum(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.mask((down == 0) & (up > 0), 100)
    return out


def macd(series):
    dif = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)
    return dif, dea, hist


def calc_atr(hist, period=14):
    high, low, close = hist["high"], hist["low"], hist["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def limit_rate(code, name):
    name = str(name).upper()
    if "ST" in name:
        return 5.0
    if code.startswith(("300", "301", "688", "689")):
        return 20.0
    return 10.0


def get_price_position(hist, period):
    """计算价格分位：(当前价 - 区间最低) / (区间最高 - 区间最低) * 100"""
    recent = hist.tail(period)
    h_max, l_min = recent["high"].max(), recent["low"].min()
    price = hist["close"].iloc[-1]
    if h_max == l_min:
        return 50.0
    return (price - l_min) / (h_max - l_min) * 100


def analyze_one(stock, config):
    code = str(stock["code"])
    name = str(stock["name"])
    price = fnum(stock.get("price"))
    pct = fnum(stock.get("change_pct"))
    high = fnum(stock.get("high"))
    low = fnum(stock.get("low"))
    open_p = fnum(stock.get("open"))
    
    if not np.isfinite(price) or price <= 0 or not np.isfinite(pct):
        return None
    if price > float(config["screening"]["max_price"]) or pct <= -10:
        return None
    limit = limit_rate(code, name)
    if pct >= limit - 0.2:
        return None

    hist = get_history(code, config)
    if len(hist) < 60:
        return None

    close = hist["close"]
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    dif, dea, mh = macd(close)
    day_rsi = rsi(close, 14)
    atr = calc_atr(hist, 14)
    
    # 基础数据
    cur_rsi = fnum(day_rsi.iloc[-1])
    cur_atr = fnum(atr.iloc[-1])
    pos20 = get_price_position(hist, 20)
    pos60 = get_price_position(hist, 60)
    vol5 = fnum(hist["volume"].tail(5).mean())
    vol20 = fnum(hist["volume"].tail(20).mean())
    vol_ratio_5_20 = vol5 / vol20 if vol20 > 0 else 1.0
    
    # 支撑压力
    recent_low = fnum(hist["low"].tail(20).min())
    recent_high = fnum(hist["high"].tail(20).max())
    dist_support = (price / recent_low - 1) * 100 if recent_low > 0 else 99
    dist_resistance = (recent_high / price - 1) * 100 if price > 0 else 0
    
    # 尾盘信号（当前快照的日内位置）
    intraday_pos = (price - low) / (high - low) * 100 if high > low else 50

    # ================= 核心 100 分制评分 =================
    score_a = 0 # 低位回调路线
    score_b = 0 # 趋势启动路线
    risk_deduction = 0 # 风险扣分

    # 1. 低位与回调质量 (满分25)
    if pos20 <= 20: score_a += 5
    elif pos20 <= 40: score_a += 3
    if pos60 <= 30: score_a += 5
    elif pos60 <= 50: score_a += 3
    if 3 <= dist_resistance <= 15: score_a += 5 # 从近期高点回落适中
    elif dist_resistance < 3: score_a += 2
    if vol_ratio_5_20 < 1.0: score_a += 5 # 缩量回调
    if 0 <= dist_support <= 3: score_a += 5 # 靠近支撑

    # 2. 趋势启动信号 (满分20)
    if price > ma5.iloc[-1] and ma5.iloc[-1] > ma5.iloc[-2]: score_b += 5
    if dif.iloc[-1] > dea.iloc[-1] or (dif.iloc[-1] < dea.iloc[-1] and mh.iloc[-1] > mh.iloc[-2]): score_b += 5
    if (hist["high"].tail(10).max() / hist["low"].tail(10).min() - 1) < 0.15: score_b += 5 # 横盘
    if 1.1 <= vol_ratio_5_20 <= 2.5: score_b += 5 # 温和放量

    # 3. 资金与量价 (满分20，内外盘占5分，成交额活跃5分，量价配合10分)
    amount_val = fnum(stock.get("amount"))
    if amount_val and stock.get("source") == "tencent": amount_val *= 10000 # 腾讯万元转元
    if amount_val and amount_val > 1e8: score_cash = 5
    else: score_cash = 0
    
    outer, inner = fnum(stock.get("outer")), fnum(stock.get("inner"))
    if np.isfinite(outer) and np.isfinite(inner) and outer > inner: score_cash += 5
    elif np.isfinite(outer) and np.isfinite(inner) and outer <= inner: score_cash += 0
    else: score_cash += 2.5 # 数据缺失给一半
    
    if pct > 0 and vol_ratio_5_20 > 1.0: score_cash += 10 # 放量上涨
    elif pct <= 0 and vol_ratio_5_20 < 1.0: score_cash += 10 # 缩量下跌
    else: score_cash += 3

    # 4. 技术趋势 (满分15)
    score_trend = 0
    if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]: score_trend += 5
    if dif.iloc[-1] > dea.iloc[-1]: score_trend += 5
    if 40 <= cur_rsi <= 65: score_trend += 5

    # 5. 尾盘买入条件 (满分15)
    score_tail = 0
    if intraday_pos >= 70: score_tail += 5 # 收盘处于日内上部
    if price >= open_p: score_tail += 5 # 收阳或平盘
    if (high - price) / price < 0.01: score_tail += 5 # 临近收盘没有明显跳水

    # 6. 风险控制 (最高扣5分)
    if pos60 > 85: risk_deduction += 2 # 高位
    if cur_rsi > 80: risk_deduction += 1 # 过热
    if dist_resistance <= 1: risk_deduction += 1 # 压力近
    if vol_ratio_5_20 > 4: risk_deduction += 1 # 异常放量

    # 汇总路线
    route_score = max(score_a, score_b)
    route_name = "低位回调型" if score_a >= score_b else "趋势启动型"
    
    total_score = route_score + score_cash + score_trend + score_tail - risk_deduction
    total_score = max(0, min(100, total_score))

    # ATR 止损
    atr_stop = price - 1.5 * cur_atr if np.isfinite(cur_atr) else price * 0.98
    stop = max(atr_stop, recent_low * 0.995) # 不跌破支撑
    if stop >= price: stop = price * 0.98

    return {
        "代码": code, "名称": name, "最新价": round(price, 2),
        "当日涨幅%": round(pct, 2), "评分": round(total_score, 1),
        "路线": route_name,
        "形态分": f"{round(route_score,1)}/45",
        "资金分": f"{round(score_cash,1)}/20",
        "技术分": f"{round(score_trend,1)}/15",
        "尾盘分": f"{round(score_tail,1)}/15",
        "扣分": f"-{risk_deduction}",
        "日线RSI": round(cur_rsi, 1), "价格分位20日": round(pos20, 1), "价格分位60日": round(pos60, 1),
        "支撑位": round(recent_low, 2), "压力位": round(recent_high, 2),
        "参考止损价": round(stop, 2),
        "止盈价+1%": round(price * 1.01, 2),
        "止盈价+2%": round(price * 1.02, 2),
    }


def market_stats(snapshot):
    df = pd.DataFrame(snapshot)
    pct = pd.to_numeric(df["change_pct"], errors="coerce")
    up = int((pct > 0).sum())
    down = int((pct < 0).sum())
    flat = int((pct == 0).sum())
    breadth = up / max(1, len(pct.dropna()))
    
    # 修正成交额单位
    amounts = []
    for s in snapshot:
        a = fnum(s.get("amount"), 0)
        if s.get("source") == "tencent": a *= 10000
        amounts.append(a)
    amount = sum(amounts)

    if breadth >= .65: mood = "偏热"
    elif breadth >= .52: mood = "温和偏强"
    elif breadth <= .35: mood = "偏冷"
    else: mood = "分化"

    return {
        "总数": len(df), "上涨": up, "下跌": down, "平盘": flat,
        "上涨比例%": round(breadth * 100, 1),
        "成交额亿元": round(amount / 1e8, 1) if amount else None,
        "市场热度": mood
    }


def load_previous():
    today = now_cn().strftime("%Y-%m-%d")
    files = []
    for fn in os.listdir(REPORT_DIR):
        match = re.fullmatch(r"picks_(\d{4}-\d{2}-\d{2})\.csv", fn)
        if match and match.group(1) < today:
            files.append((match.group(1), fn))
    if not files: return None, pd.DataFrame()
    previous_date, fn = sorted(files)[-1]
    try:
        return previous_date, pd.read_csv(os.path.join(REPORT_DIR, fn), dtype={"代码": str})
    except Exception:
        return None, pd.DataFrame()


def review_previous(snapshot):
    date, old = load_previous()
    if old.empty: return date, pd.DataFrame()

    current = {x["code"]: x for x in snapshot}
    rows = []

    for _, item in old.iterrows():
        code = str(item.get("代码", "")).zfill(6)
        cur = current.get(code)
        if not cur: continue

        old_price = fnum(item.get("最新价"))
        new_price = fnum(cur.get("price"))
        change = (new_price / old_price - 1) * 100 if old_price > 0 and new_price > 0 else np.nan

        if not np.isfinite(change): reason = "价格数据不完整"
        elif change <= -2: reason = "明显下跌：需检查市场环境、支撑失守"
        elif change < 0: reason = "小幅回撤：观察支撑位和开盘承接"
        else: reason = "价格未下跌，记录正向样本"

        rows.append({
            "推荐日期": date, "代码": code, "名称": item.get("名称"),
            "前日参考价": old_price, "当前价": new_price,
            "T+1变化%": round(change, 2) if np.isfinite(change) else None,
            "复盘结论": reason, "前日评分": item.get("评分"), "路线": item.get("路线", "未知")
        })
    return date, pd.DataFrame(rows)


def save_learning(review):
    if review.empty: return
    path = os.path.join(REPORT_DIR, "overnight_learning.csv")
    if os.path.exists(path):
        try:
            old = pd.read_csv(path, dtype={"代码": str})
            review = pd.concat([old, review], ignore_index=True)
        except Exception: pass
    review = review.drop_duplicates(["推荐日期", "代码"], keep="last")
    review.to_csv(path, index=False, encoding="utf-8-sig")


def write_report(day, stats, picks, candidates, review, source_status):
    lines = [
        f"# A股猎手 V4｜隔夜超短报告", f"**日期：{day}**", "",
        "## 01｜市场温度", f"**{stats['市场热度']}**　上涨 {stats['上涨']} / 下跌 {stats['下跌']} / 平盘 {stats['平盘']}",
        f"两市成交额估算：{stats['成交额亿元']} 亿元", "",
        "## 02｜隔夜综合排名前三"
    ]

    if picks.empty:
        lines.append("本轮无有效候选。")
    else:
        for i, (_, x) in enumerate(picks.iterrows(), 1):
            lines += [
                "", f"### {i}. {x['名称']}（{x['代码']}）　**{x['评分']}分** (路线: {x['路线']})",
                f"- 形态分: {x['形态分']}　资金分: {x['资金分']}　技术分: {x['技术分']}　尾盘分: {x['尾盘分']}　风险扣分: {x['扣分']}",
                f"- 现价：**{x['最新价']}元**　涨幅：{x['当日涨幅%']}%",
                f"- 止盈：**{x['止盈价+1%']} / {x['止盈价+2%']}**　参考止损：**{x['参考止损价']}**",
                f"- 支撑：{x['支撑位']}　压力：{x['压力位']}　20日/60日分位：{x['价格分位20日']}% / {x['价格分位60日']}%"
            ]

    lines += ["", "## 03｜候选名单 TOP 30"]
    if not candidates.empty:
        lines.append("|排名|代码|名称|评分|路线|涨幅|止盈1%|止损|")
        lines.append("|---:|---|---|---:|---|---:|---:|---:|")
        for i, (_, x) in enumerate(candidates.head(30).iterrows(), 1):
            lines.append(f"|{i}|{x['代码']}|{x['名称']}|{x['评分']}|{x['路线']}|{x['当日涨幅%']}%|{x['止盈价+1%']}|{x['参考止损价']}|")

    lines += ["", "## 04｜上一轮复盘"]
    if review.empty:
        lines.append("暂无可复盘的上一轮推荐。")
    else:
        for _, x in review.iterrows():
            lines.append(f"- {x['名称']}：变化 {x['T+1变化%']}%；{x['复盘结论']}")

    with open(os.path.join(REPORT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run():
    start = time.time()
    day = now_cn().strftime("%Y-%m-%d")

    reset_data_source_status()
    log("获取行情快照……")
    snapshot = get_market_snapshot(CONFIG)
    stats = market_stats(snapshot)

    screen = []
    for s in snapshot:
        code, name = s["code"], s["name"]
        price = fnum(s.get("price"))
        pct = fnum(s.get("change_pct"))
        if not np.isfinite(price) or not np.isfinite(pct): continue
        if price <= 0 or price > float(CONFIG["screening"]["max_price"]): continue
        if pct <= -10 or pct >= limit_rate(code, name) - 0.2: continue
        screen.append(s)

    log(f"过滤后：{len(screen)}只，开始 V4 深度评分。")
    workers = int(CONFIG.get("data_source", {}).get("max_workers", 8))
    analyzed = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(analyze_one, s, CONFIG): s for s in screen}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                item = future.result()
                if item: analyzed.append(item)
            except Exception:
                pass
            if i % 200 == 0: log(f"进度 {i}/{len(futures)}，有效 {len(analyzed)}")

    df = pd.DataFrame(analyzed)
    if not df.empty:
        df = df.sort_values(["评分", "当日涨幅%"], ascending=[False, False]).reset_index(drop=True)
        df.to_csv(os.path.join(REPORT_DIR, f"candidates_{day}.csv"), index=False, encoding="utf-8-sig")

    picks = df.head(3).copy() if not df.empty else pd.DataFrame()
    if not picks.empty:
        picks.to_csv(os.path.join(REPORT_DIR, f"picks_{day}.csv"), index=False, encoding="utf-8-sig")

    prev_date, review = review_previous(snapshot)
    save_learning(review)
    write_report(day, stats, picks, df, review, get_data_source_status())

    log(f"完成：候选{len(df)}只，前三名{len(picks)}只。")


if __name__ == "__main__":
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            CONFIG = json.load(f)
        run()
    except Exception as exc:
        log(f"扫描失败：{type(exc).__name__}: {exc}")
        traceback.print_exc()
