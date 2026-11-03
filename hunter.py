# -*- coding: utf-8 -*-
"""A股猎手：14:50尾盘隔夜超短线扫描（融合技术指标与资金动能）"""
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

def limit_rate(code, name):
    name = str(name).upper()
    if "ST" in name: return 5.0
    if code.startswith(("300", "301", "688", "689")): return 20.0
    return 10.0

def support_resistance(hist, price):
    low_col = hist["low"]
    high_col = hist["high"]
    ma20 = fnum(hist["close"].rolling(20).mean().iloc[-1])
    ma60 = fnum(hist["close"].rolling(60).mean().iloc[-1])
    lows = [fnum(low_col.tail(10).min()), fnum(low_col.tail(20).min()), fnum(low_col.tail(60).min()), ma20, ma60]
    highs = [fnum(high_col.tail(10).max()), fnum(high_col.tail(20).max()), fnum(high_col.tail(60).max())]
    supports = sorted([x for x in lows if np.isfinite(x) and 0 < x <= price])
    resistances = sorted([x for x in highs if np.isfinite(x) and x >= price])
    return (supports[-1] if supports else np.nan), (resistances[0] if resistances else np.nan)

def analyze_one(stock, config):
    code, name = str(stock["code"]), str(stock["name"])
    price, pct = fnum(stock.get("price")), fnum(stock.get("change_pct"))
    high, low = fnum(stock.get("high")), fnum(stock.get("low"))
    outer, inner = fnum(stock.get("outer"), 0), fnum(stock.get("inner"), 0)
    
    if not np.isfinite(price) or price <= 0 or not np.isfinite(pct): return None
    if price > float(config["screening"]["max_price"]): return None
    if pct <= -10: return None
    if pct >= limit_rate(code, name) - 0.2: return None

    hist = get_history(code, config)
    if hist is None or len(hist) < 60: return None

    close = hist["close"]
    ma5, ma10, ma20 = close.rolling(5).mean(), close.rolling(10).mean(), close.rolling(20).mean()
    dif, dea, mh = macd(close)
    
    support, resistance = support_resistance(hist, price)
    
    score = 0.0
    reasons, risks = [], []

    # 1. 资金抢筹形态 (3分)
    day_range = high - low
    if day_range > 0:
        close_position = (price - low) / day_range
        if close_position >= 0.95:
            score += 3.0
            reasons.append("尾盘收于最高点(主力抢筹)")
        elif close_position >= 0.8:
            score += 1.5
            reasons.append("尾盘走势强劲")
        elif close_position < 0.5:
            risks.append("尾盘回落明显，资金承接弱")

    # 2. 主动资金买入 (2分)
    if outer > inner and inner > 0:
        score += 2.0
        reasons.append("外盘>内盘(主动买盘强)")
    elif inner > outer * 1.5:
        risks.append("主动卖盘巨大")

    # 3. 量能配合 (1.5分)
    volume = fnum(hist["volume"].iloc[-1])
    avg_volume = fnum(hist["volume"].tail(5).mean())
    volume_ratio = volume / avg_volume if avg_volume > 0 else np.nan
    if np.isfinite(volume_ratio) and 1.2 <= volume_ratio <= 3.0:
        score += 1.5
        reasons.append(f"温和放量({volume_ratio:.1f}倍)")
    elif np.isfinite(volume_ratio) and volume_ratio > 4.5:
        risks.append("天量分歧，谨防出货")

    # 4. 技术指标安全垫 (1.5分)
    if price > fnum(ma5.iloc[-1]) and price > fnum(ma10.iloc[-1]):
        score += 1.0
        reasons.append("站上5/10日线")
    else:
        risks.append("跌破均线支撑")

    if fnum(dif.iloc[-1]) > fnum(dea.iloc[-1]) or fnum(mh.iloc[-1]) > fnum(mh.iloc[-2]):
        score += 0.5
        reasons.append("日线MACD动能向好")

    # 5. 涨幅区间 (2分)
    if 1.0 <= pct <= 4.5:
        score += 2.0
        reasons.append("涨幅处于最佳上攻区间")
    elif -1.0 <= pct < 0.2 and price >= fnum(ma20.iloc[-1]):
        score += 1.0
        reasons.append("强势趋势内小幅整理")
    elif pct > 6.0:
        risks.append("涨幅偏大，隔夜追高风险")

    if price < fnum(ma20.iloc[-1]) or price < support:
        return None

    score = max(0, min(10, score))

    tp1, tp2 = round(price * 1.005, 2), round(price * 1.015, 2)
    stop_loss = round(max(low * 0.99, fnum(ma5.iloc[-1]) * 0.99, fnum(ma10.iloc[-1]) * 0.99), 2)
    if stop_loss >= price: stop_loss = round(price * 0.98, 2)

    return {
        "代码": code, "名称": name, "最新价": round(price, 2),
        "当日涨幅%": round(pct, 2), "评分": round(score, 2),
        "止盈目标1": tp1, "止盈目标2": tp2, "止损价": stop_loss,
        "支撑位": round(support, 2) if np.isfinite(support) else None,
        "压力位": round(resistance, 2) if np.isfinite(resistance) else None,
        "资金逻辑": "；".join(reasons) if reasons else "无强资金驱动",
        "风险提示": "；".join(risks) if risks else "无明显风险",
    }

def market_stats(snapshot):
    df = pd.DataFrame(snapshot)
    pct = pd.to_numeric(df["change_pct"], errors="coerce")
    breadth = int((pct > 0).sum()) / max(1, len(pct.dropna()))
    if breadth >= 0.6: mood = "🔥 极热"
    elif breadth >= 0.45: mood = "🌤 温和"
    else: mood = "❄️ 偏冷"
    return {"上涨比例%": round(breadth * 100, 1), "市场情绪": mood}

def load_previous():
    today = now_cn().strftime("%Y-%m-%d")
    files = []
    for fn in os.listdir(REPORT_DIR):
        match = re.fullmatch(r"picks_(\d{4}-\d{2}-\d{2})\.csv", fn)
        if match and match.group(1) < today: files.append((match.group(1), fn))
    if not files: return None, pd.DataFrame()
    previous_date, fn = sorted(files)[-1]
    try:
        old = pd.read_csv(os.path.join(REPORT_DIR, fn), dtype={"代码": str})
        return previous_date, old
    except Exception: return None, pd.DataFrame()

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
        tp1 = fnum(item.get("止盈目标1"))
        sl = fnum(item.get("止损价"))
        
        open_p = fnum(cur.get("open"))
        high_p = fnum(cur.get("high"))
        low_p = fnum(cur.get("low"))
        close_p = fnum(cur.get("price"))
        
        # 次日表现归因分析
        if not np.isfinite(close_p): 
            continue
            
        change = (close_p / old_price - 1) * 100 if old_price > 0 else np.nan
        
        if change >= 1.0:
            reason = f"✅ 大肉！次日大涨{change:.2f}%，冲高止盈"
        elif change > 0:
            reason = f"✅ 微利离场，次日红盘{change:.2f}%"
        elif change > -1.5:
            reason = f"⚪ 平盘/微亏{change:.2f}%，正常震荡，纪律离场"
        else:
            # 深度归因（说明选股判断错误）
            if high_p >= tp1 and close_p < old_price:
                reason = f"❌ 贪心被套！次日最高曾达{tp1}，但收盘跌破买入价"
            elif open_p < sl and low_p < sl:
                reason = f"❌ 破位杀跌！直接低开跌破止损线{sl}"
            elif fnum(cur.get("change_pct"), 0) < -1.5:
                reason = f"❌ 选股失误！跟随大盘/板块跳水，跌幅{change:.2f}%"
            else:
                reason = f"❌ 弱势阴跌！跌破止损线{sl}，认栽出局"

        rows.append({
            "推荐日期": date, "代码": code, "名称": item.get("名称"),
            "前日买入价": old_price, "今日现价": close_p,
            "昨日止盈目标": tp1, "昨日止损底线": sl,
            "T+1变化%": round(change, 2) if np.isfinite(change) else None,
            "复盘归因": reason
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
    # 高密度 PPT 卡片式报告
    lines = [
        f"# 🎯 A股猎手｜隔夜超短报告 ({day})",
        f"⏰ **时间**：{now_cn():%H:%M}　|　**市场温度**：{stats['市场情绪']} (上涨 {stats['上涨比例%']}%)",
        f"💡 **纪律**：14:50 买入，次日 09:30-09:40 冲高即走，不格局。",
        "---",
        "## 🏆 今日狙击目标 (按评分排序)"
    ]

    if picks.empty:
        lines.append("> ⚠️ 今日无符合尾盘强势特征的标的，建议空仓休息。")
    else:
        for i, (_, x) in enumerate(picks.iterrows(), 1):
            lines += [
                f"### {i}️⃣ {x['名称']} ({x['代码']}) — 评分: `{x['评分']}/10`",
                f"> 👉 **尾盘买入价**：**{x['最新价']}** 元 (涨幅 {x['当日涨幅%']}%)",
                f"> 🎯 **次日止盈**：**{x['止盈目标1']}** (+0.5%) / **{x['止盈目标2']}** (+1.5%)",
                f"> 🛡️ **防守底线**：跌破 **{x['止损价']}** 无条件止损！",
                f"- 🔥 **资金逻辑**：{x['资金逻辑']}",
                f"- ⚠️ **风险预警**：{x['风险提示']}",
                ""
            ]

    lines += [
        "---",
        "## 📊 备选候选池 (前10名)",
        "| 代码 | 名称 | 评分 | 涨幅 | 现价 | 止盈+0.5% | 防守止损 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    if not candidates.empty:
        for _, x in candidates.head(10).iterrows():
            lines.append(f"| {x['代码']} | {x['名称']} | **{x['评分']}** | {x['当日涨幅%']}% | {x['最新价']} | {x['止盈目标1']} | {x['止损价']} |")
    
    lines += [
        "---",
        f"## 🔍 上一交易日复盘 (基于 {review.iloc[0]['推荐日期'] if not review.empty else '历史'} 推荐)",
    ]
    if review.empty:
        lines.append("> 暂无历史战绩记录。")
    else:
        for _, x in review.iterrows():
            lines += [
                f"**{x['名称']} ({x['代码']})**：昨日买入价 {x['前日买入价']} ➔ 今日收盘 {x['今日现价']} ( **{x['T+1变化%']}%** )",
                f"- 🎯 昨日计划止盈：{x['昨日止盈目标']} / 🛡️ 计划止损：{x['昨日止损底线']}",
                f"- 💡 **归因**：{x['复盘归因']}",
                ""
            ]

    lines += [
        "---",
        "> **🛑 铁血纪律**：超短线博弈本质是概率游戏。若次日 9:30-9:40 未能冲高，平盘或微亏务必坚决离场，绝不把超短做成长线套牢！"
    ]

    with open(os.path.join(REPORT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def run():
    start = time.time()
    day = now_cn().strftime("%Y-%m-%d")

    reset_data_source_status()
    log("获取全市场实时行情……")
    snapshot = get_market_snapshot(CONFIG)
    stats = market_stats(snapshot)

    screen = []
    for s in snapshot:
        code, name = s["code"], s["name"]
        price, pct = fnum(s.get("price")), fnum(s.get("change_pct"))

        if not np.isfinite(price) or not np.isfinite(pct): continue
        if price <= 0 or price > float(CONFIG["screening"]["max_price"]): continue
        if pct <= -10: continue
        if pct >= limit_rate(code, name) - 0.2: continue

        screen.append(s)

    log(f"硬性过滤后：{len(screen)}只，开始技术分析。")

    workers = int(CONFIG.get("data_source", {}).get("max_workers", 8))
    analyzed, errors = [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(analyze_one, s, CONFIG): s for s in screen}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                item = future.result()
                if item: analyzed.append(item)
            except Exception as exc:
                errors.append({"代码": futures[future]["code"], "名称": futures[future]["name"], "错误": str(exc)[:300]})

            if i % 200 == 0 or i == len(futures):
                log(f"进度 {i}/{len(futures)}，有效分析 {len(analyzed)}")

    df = pd.DataFrame(analyzed)

    if not df.empty:
        df = df.sort_values(["评分", "当日涨幅%"], ascending=[False, True]).reset_index(drop=True)

    picks = df.head(3).copy() if not df.empty else pd.DataFrame()

    if not df.empty:
        df.to_csv(os.path.join(REPORT_DIR, f"candidates_{day}.csv"), index=False, encoding="utf-8-sig")
    if not picks.empty:
        picks.to_csv(os.path.join(REPORT_DIR, f"picks_{day}.csv"), index=False, encoding="utf-8-sig")

    prev_date, review = review_previous(snapshot)
    save_learning(review)

    if errors:
        pd.DataFrame(errors).to_csv(
            os.path.join(REPORT_DIR, "scan_errors.csv"),
            mode="a", header=not os.path.exists(os.path.join(REPORT_DIR, "scan_errors.csv")),
            index=False, encoding="utf-8-sig"
        )

    source_status = get_data_source_status()
    write_report(day, stats, picks, df, review, source_status)

    log(f"完成：候选{len(df)}只，前三名{len(picks)}只，错误{len(errors)}条。")
    log("报告位置：reports/report.md")

if __name__ == "__main__":
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            CONFIG = json.load(f)
        run()
    except Exception as exc:
        log(f"扫描失败：{type(exc).__name__}: {exc}")
        traceback.print_exc()
        with open(os.path.join(REPORT_DIR, "fatal_error.log"), "a", encoding="utf-8") as f:
            f.write(f"\n{now_cn().isoformat()} {type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc())
        raise
