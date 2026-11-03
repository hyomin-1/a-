# -*- coding: utf-8 -*-
"""A股猎手：14:50尾盘隔夜超短线扫描（极简版）"""
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

def limit_rate(code, name):
    name = str(name).upper()
    if "ST" in name:
        return 5.0
    if code.startswith(("300", "301", "688", "689")):
        return 20.0
    return 10.0

def analyze_one(stock, config):
    code = str(stock["code"])
    name = str(stock["name"])
    price = fnum(stock.get("price"))
    pct = fnum(stock.get("change_pct"))
    high = fnum(stock.get("high"))
    low = fnum(stock.get("low"))
    
    # 基础过滤：不追涨停、不买跌停、价格过高不买
    if not np.isfinite(price) or price <= 0 or not np.isfinite(pct): return None
    if pct >= limit_rate(code, name) - 0.2: return None # 涨停排除
    if pct <= -9.8: return None # 跌停排除
    
    hist = get_history(code, config)
    if hist is None or len(hist) < 20: return None
    
    close = hist["close"]
    ma5 = fnum(close.rolling(5).mean().iloc[-1])
    ma10 = fnum(close.rolling(10).mean().iloc[-1])
    ma20 = fnum(close.rolling(20).mean().iloc[-1])
    
    # 计算近20日最低点作为大支撑
    low20 = fnum(hist["low"].tail(20).min())
    
    # 量价与资金意图
    volume = fnum(hist["volume"].iloc[-1])
    avg_vol_5 = fnum(hist["volume"].tail(5).mean())
    vol_ratio = volume / avg_vol_5 if avg_vol_5 > 0 else 1.0
    
    # 外盘内盘（代表主动买卖）
    outer = fnum(stock.get("outer"), 0)
    inner = fnum(stock.get("inner"), 0)

    # ================= 核心打分 (满分10分) =================
    score = 0.0
    reasons, risks = [], []

    # 1. 尾盘形态 (3分)：收盘价越接近全天最高价，说明尾盘资金抢筹，次日高开概率大
    day_range = high - low
    if day_range > 0:
        close_position = (price - low) / day_range
        if close_position >= 0.95:
            score += 3.0
            reasons.append("尾盘收于最高点附近(抢筹明显)")
        elif close_position >= 0.8:
            score += 1.5
            reasons.append("尾盘走势较强")
        elif close_position < 0.5:
            risks.append("尾盘回落明显，资金承接弱")

    # 2. 量能指标 (2分)：量比温和放大，说明有资金介入但非天量出货
    if 1.2 <= vol_ratio <= 3.0:
        score += 2.0
        reasons.append(f"成交量温和放大(量比{vol_ratio:.1f})")
    elif vol_ratio > 4.5:
        risks.append("成交量异常巨大，防范出货")

    # 3. 涨幅区间 (2分)：超短线最忌讳追高，1%-4%之间最具爆发力
    if 1.0 <= pct <= 4.0:
        score += 2.0
        reasons.append("当日涨幅处于最佳上攻区间")
    elif pct > 6.0:
        risks.append("当日涨幅过大，次日容易低开")

    # 4. 资金主动性 (1分)
    if outer > inner and inner > 0:
        score += 1.0
        reasons.append("主动买盘 > 主动卖盘")

    # 5. 支撑与防守 (2分)
    if price > ma5:
        score += 1.0
        reasons.append("站上5日线")
    if price > ma10:
        score += 1.0
    else:
        risks.append("跌破10日线，趋势转弱")

    score = max(0, min(10, score))
    
    # 如果跌破20日大支撑，直接放弃
    if price < low20 or price < ma20:
        return None

    # ================= 交易计划 (价格设定) =================
    # 止盈：给出 0.5% 和 1.5% 两档目标
    tp1 = round(price * 1.005, 2)
    tp2 = round(price * 1.015, 2)
    
    # 止损：放在今日最低点或 MA5 之下
    stop_loss = round(max(low * 0.99, ma5 * 0.99), 2)
    if stop_loss >= price:
        stop_loss = round(price * 0.98, 2)

    return {
        "代码": code, "名称": name, "最新价": round(price, 2),
        "当日涨幅%": round(pct, 2), "评分": round(score, 2),
        "止盈目标1": tp1, "止盈目标2": tp2, "止损价": stop_loss,
        "评分理由": "；".join(reasons) if reasons else "无强驱动",
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
        if match and match.group(1) < today:
            files.append((match.group(1), fn))
    if not files: return None, pd.DataFrame()
    previous_date, fn = sorted(files)[-1]
    try:
        old = pd.read_csv(os.path.join(REPORT_DIR, fn), dtype={"代码": str})
        return previous_date, old
    except: return None, pd.DataFrame()

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
        if not np.isfinite(change): reason = "数据缺失"
        elif change >= 1.0: reason = "高开止盈成功"
        elif change > 0: reason = "微利离场"
        elif change > -1.0: reason = "平盘附近"
        else: reason = "⚠️ 破位亏损"
        
        rows.append({"代码": code, "名称": item.get("名称"), "T+1变化%": round(change, 2) if np.isfinite(change) else None, "结果": reason})
    return date, pd.DataFrame(rows)

def write_report(day, stats, picks, candidates, review):
    # PPT 风格卡片式报告
    lines = [
        f"# 🎯 A股猎手｜隔夜超短报告 ({day})",
        f"⏰ **运行时间**：{now_cn():%H:%M}　|　**市场情绪**：{stats['市场情绪']} (上涨占比 {stats['上涨比例%']}%)",
        f"💡 **操盘纪律**：14:50 附近择机买入，次日 09:30 - 09:40 冲高即走，持仓不超过 1 夜。",
        "---",
        "## 🏆 精选尾盘金股（按评分排序）"
    ]

    if picks.empty:
        lines.append("> ⚠️ 今日无符合尾盘强势特征的标的，建议空仓等待。")
    else:
        for i, (_, x) in enumerate(picks.iterrows(), 1):
            lines += [
                f"### {i}️⃣ {x['名称']} ({x['代码']}) —— `评分 {x['评分']}/10`",
                f"- 💰 **买入参考**：**{x['最新价']}** 元　|　📈 **今日涨幅**：{x['当日涨幅%']}%",
                f"- 🎯 **次日止盈**：+0.5% ( **{x['止盈目标1']}** ) / +1.5% ( **{x['止盈目标2']}** )",
                f"- 🛡️ **防守底线**：跌破 **{x['止损价']}** 无条件止损 (微亏出局保平安)",
                f"- 💡 **资金逻辑**：{x['评分理由']}",
                f"- ⚠️ **风险提示**：{x['风险提示']}",
                ""
            ]

    lines += [
        "---",
        "## 📊 全市场候选池",
        f"经过全市场扫描，符合基础过滤的标的共 **{len(candidates)}** 只，以下是排名前 10 的备选：",
        "| 代码 | 名称 | 评分 | 涨幅 | 现价 | 建议止盈 | 建议止损 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    if not candidates.empty:
        for _, x in candidates.head(10).iterrows():
            lines.append(f"| {x['代码']} | {x['名称']} | **{x['评分']}** | {x['当日涨幅%']}% | {x['最新价']} | {x['止盈目标1']} | {x['止损价']} |")
    
    lines += [
        "---",
        "## 🔍 上一轮复盘",
    ]
    if review.empty:
        lines.append("> 暂无历史推荐记录。")
    else:
        for _, x in review.iterrows():
            lines.append(f"- {x['名称']} ({x['代码']})：次日表现 **{x['T+1变化%']}%** ( {x['结果']} )")

    lines += [
        "---",
        "> **🛑 风险提示**：超短线博弈胜率与执行纪律高度相关，若次日低开且跌破防守底线，请果断离场，切勿扛单。本工具仅供技术筛选参考。"
    ]

    with open(os.path.join(REPORT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def run():
    start = time.time()
    day = now_cn().strftime("%Y-%m-%d")
    reset_data_source_status()
    
    log("正在获取全市场行情...")
    snapshot = get_market_snapshot(CONFIG)
    stats = market_stats(snapshot)

    # 初筛：剔除科创、北交、ST、停牌，且不追当日涨停
    screen = []
    for s in snapshot:
        code, name = s["code"], s["name"]
        price = fnum(s.get("price"))
        pct = fnum(s.get("change_pct"))
        
        if not np.isfinite(price) or not np.isfinite(pct): continue
        if code.startswith(("688", "8", "4", "9")) or "ST" in name.upper(): continue # 核心过滤
        if pct >= limit_rate(code, name) - 0.2: continue # 过滤涨停

        screen.append(s)

    log(f"过滤后剩余 {len(screen)} 只，开始深度技术分析...")

    analyzed = []
    workers = int(CONFIG.get("data_source", {}).get("max_workers", 10))
    
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(analyze_one, s, CONFIG): s for s in screen}
        for i, future in enumerate(as_completed(futures), 1):
            item = future.result()
            if item: analyzed.append(item)
            if i % 200 == 0: log(f"已完成 {i} 只标的扫描...")

    df = pd.DataFrame(analyzed)
    if not df.empty:
        # 按评分降序排列，评分相同的按涨幅小的排前面（防追高）
        df = df.sort_values(["评分", "当日涨幅%"], ascending=[False, True]).reset_index(drop=True)

    picks = df.head(3).copy() if not df.empty else pd.DataFrame()

    if not df.empty:
        df.to_csv(os.path.join(REPORT_DIR, f"candidates_{day}.csv"), index=False, encoding="utf-8-sig")
    if not picks.empty:
        picks.to_csv(os.path.join(REPORT_DIR, f"picks_{day}.csv"), index=False, encoding="utf-8-sig")

    prev_date, review = review_previous(snapshot)
    write_report(day, stats, picks, df, review)
    log(f"完成！耗时 {round(time.time() - start, 1)} 秒。请查看 reports/report.md")

if __name__ == "__main__":
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            CONFIG = json.load(f)
        run()
    except Exception as exc:
        log(f"扫描失败：{type(exc).__name__}: {exc}")
        traceback.print_exc()
