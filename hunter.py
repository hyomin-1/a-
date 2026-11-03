# -*- coding: utf-8 -*-
"""A股猎手：14:50尾盘隔夜超短线扫描与T+1复盘。"""
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
import requests

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
    if "ST" in name:
        return 5.0
    if code.startswith(("300", "301", "688", "689")):
        return 20.0
    return 10.0


def support_resistance(hist, price):
    """用近10/20/60日高低点及MA20/MA60估算支撑压力。"""
    low_col = hist["low"]
    high_col = hist["high"]
    ma20 = fnum(hist["close"].rolling(20).mean().iloc[-1])
    ma60 = fnum(hist["close"].rolling(60).mean().iloc[-1])

    lows = [
        fnum(low_col.tail(10).min()),
        fnum(low_col.tail(20).min()),
        fnum(low_col.tail(60).min()),
        ma20, ma60
    ]
    highs = [
        fnum(high_col.tail(10).max()),
        fnum(high_col.tail(20).max()),
        fnum(high_col.tail(60).max())
    ]

    supports = sorted([x for x in lows if np.isfinite(x) and 0 < x <= price])
    resistances = sorted([x for x in highs if np.isfinite(x) and x >= price])

    support = supports[-1] if supports else np.nan
    resistance = resistances[0] if resistances else np.nan
    return support, resistance, ma20, ma60


def analyze_one(stock, config):
    code = str(stock["code"])
    name = str(stock["name"])
    price = fnum(stock.get("price"))
    pct = fnum(stock.get("change_pct"))
    if not np.isfinite(price) or price <= 0 or not np.isfinite(pct):
        return None

    if price > float(config["screening"]["max_price"]):
        return None
    if pct <= -10:
        return None
    limit = limit_rate(code, name)
    if pct >= limit - 0.2:
        return None

    hist = get_history(code, config)
    close = hist["close"]
    if len(hist) < 60:
        return None

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    dif, dea, mh = macd(close)
    day_rsi = rsi(close, 14)

    support, resistance, m20, m60 = support_resistance(hist, price)
    deviation20 = (price / m20 - 1) * 100 if m20 > 0 else np.nan

    # 评分逻辑
    score = 0.0
    reasons, risks = [], []

    if price > fnum(ma10.iloc[-1]) and fnum(ma10.iloc[-1]) >= fnum(ma10.iloc[-2]):
        score += 1.0
        reasons.append("价格站上MA10且MA10走平向上")
    elif price > fnum(ma20.iloc[-1]):
        score += 0.5
        reasons.append("价格仍在MA20上方")
    else:
        risks.append("价格位于MA20下方")

    if fnum(ma20.iloc[-1]) >= fnum(ma20.iloc[-3]):
        score += 0.5
        reasons.append("MA20趋势未明显走弱")

    if fnum(dif.iloc[-1]) > fnum(dea.iloc[-1]):
        score += 0.7
        reasons.append("日线MACD处于多头状态")
    elif fnum(mh.iloc[-1]) > fnum(mh.iloc[-2]):
        score += 0.4
        reasons.append("日线MACD柱体改善")
    else:
        risks.append("日线MACD动能偏弱")

    drsi = fnum(day_rsi.iloc[-1])
    if 45 <= drsi <= 68:
        score += 0.6
        reasons.append("日线RSI处于适中区间")
    elif drsi > 78:
        risks.append("日线RSI过热")

    # 当日涨幅适中
    if 0.2 <= pct <= 3.5:
        score += 0.7
        reasons.append("当日涨幅适中")
    elif -1.0 <= pct < 0.2 and price >= fnum(ma20.iloc[-1]):
        score += 0.4
        reasons.append("强势趋势内小幅整理")
    elif pct > 5:
        risks.append("当日涨幅偏大，隔夜追高风险")

    support_gap = (price / support - 1) * 100 if np.isfinite(support) and support > 0 else np.nan
    resistance_gap = (resistance / price - 1) * 100 if np.isfinite(resistance) and price > 0 else np.nan

    if np.isfinite(support_gap) and 0 <= support_gap <= 3:
        score += 0.8
        reasons.append("价格靠近近端支撑区域")
    elif np.isfinite(support_gap) and support_gap > 7:
        risks.append("距离下方支撑较远")

    if np.isfinite(resistance_gap) and resistance_gap <= 1.2:
        score -= 0.6
        risks.append("上方压力位较近，突破空间有限")
    elif np.isfinite(resistance_gap) and resistance_gap >= 2.5:
        score += 0.5
        reasons.append("上方有一定空间")

    # 量价
    volume = fnum(hist["volume"].iloc[-1])
    avg_volume = fnum(hist["volume"].tail(5).mean())
    volume_ratio = volume / avg_volume if avg_volume > 0 else np.nan
    if np.isfinite(volume_ratio) and 1.05 <= volume_ratio <= 2.5:
        score += 0.5
        reasons.append("近期成交量有温和活跃")
    elif np.isfinite(volume_ratio) and volume_ratio > 4:
        risks.append("成交量异常放大，注意分歧")

    # 委比/内外盘
    outer = fnum(stock.get("outer"))
    inner = fnum(stock.get("inner"))
    if np.isfinite(outer) and np.isfinite(inner) and outer + inner > 0:
        if outer > inner:
            score += 0.3
            reasons.append("外盘成交暂时强于内盘")
        else:
            risks.append("内盘成交暂时偏强")

    score = max(0, min(10, score))

    tp05 = round(price * 1.005, 2)
    tp1 = round(price * 1.01, 2)
    tp2 = round(price * 1.02, 2)

    stop_default = price * 0.992
    stop_support = support * 0.997 if np.isfinite(support) and support > 0 else np.nan
    stop = max(stop_default, stop_support) if np.isfinite(stop_support) else stop_default
    stop = round(stop, 2)

    if stop >= price:
        stop = round(stop_default, 2)

    return {
        "代码": code, "名称": name, "最新价": round(price, 2),
        "当日涨幅%": round(pct, 2), "评分": round(score, 2),
        "日线MACD": "多头" if fnum(dif.iloc[-1]) > fnum(dea.iloc[-1]) else "偏弱",
        "日线RSI": round(drsi, 1) if np.isfinite(drsi) else None,
        "MA10": round(fnum(ma10.iloc[-1]), 2),
        "MA20": round(m20, 2) if np.isfinite(m20) else None,
        "MA60": round(m60, 2) if np.isfinite(m60) else None,
        "支撑位": round(support, 2) if np.isfinite(support) else None,
        "压力位": round(resistance, 2) if np.isfinite(resistance) else None,
        "距离支撑%": round(support_gap, 2) if np.isfinite(support_gap) else None,
        "距离压力%": round(resistance_gap, 2) if np.isfinite(resistance_gap) else None,
        "最低止盈价+0.5%": tp05,
        "止盈价+1%": tp1,
        "止盈价+2%": tp2,
        "参考止损价": stop,
        "量比代理": round(volume_ratio, 2) if np.isfinite(volume_ratio) else None,
        "板块": "当前数据源未提供可靠行业映射",
        "板块资金": "不可验证，不计入评分",
        "评分理由": "；".join(reasons),
        "风险提示": "；".join(risks) if risks else "未发现明显技术风险",
        "数据完整性": "日线数据完整（关闭15分钟线防封IP）"
    }


def market_stats(snapshot):
    df = pd.DataFrame(snapshot)
    pct = pd.to_numeric(df["change_pct"], errors="coerce")
    up = int((pct > 0).sum())
    down = int((pct < 0).sum())
    flat = int((pct == 0).sum())
    up_limit = int((pct >= 9.5).sum())
    down_limit = int((pct <= -9.5).sum())
    breadth = up / max(1, len(pct.dropna()))
    amount = pd.to_numeric(df["amount"], errors="coerce").sum(min_count=1)

    if breadth >= .65 and down_limit < 15:
        mood = "偏热"
    elif breadth >= .52 and down_limit < 25:
        mood = "温和偏强"
    elif breadth <= .35 or down_limit >= 30:
        mood = "偏冷"
    else:
        mood = "分化"

    return {
        "总数": len(df), "上涨": up, "下跌": down, "平盘": flat,
        "涨停附近": up_limit, "跌停附近": down_limit,
        "上涨比例%": round(breadth * 100, 1),
        "成交额亿元": round(amount / 1e8, 1) if pd.notna(amount) else None,
        "市场热度": mood
    }


def load_previous():
    today = now_cn().strftime("%Y-%m-%d")
    files = []
    for fn in os.listdir(REPORT_DIR):
        match = re.fullmatch(r"picks_(\d{4}-\d{2}-\d{2})\.csv", fn)
        if match and match.group(1) < today:
            files.append((match.group(1), fn))
    if not files:
        return None, pd.DataFrame()

    previous_date, fn = sorted(files)[-1]
    try:
        old = pd.read_csv(os.path.join(REPORT_DIR, fn), dtype={"代码": str})
        return previous_date, old
    except Exception as exc:
        log(f"读取上一轮推荐失败：{exc}")
        return None, pd.DataFrame()


def review_previous(snapshot):
    date, old = load_previous()
    if old.empty:
        return date, pd.DataFrame()

    current = {x["code"]: x for x in snapshot}
    rows = []

    for _, item in old.iterrows():
        code = str(item.get("代码", "")).zfill(6)
        cur = current.get(code)
        if not cur:
            continue

        old_price = fnum(item.get("最新价"))
        new_price = fnum(cur.get("price"))
        change = (new_price / old_price - 1) * 100 if old_price > 0 and new_price > 0 else np.nan

        if not np.isfinite(change):
            reason = "价格数据不完整"
        elif change <= -2:
            reason = "明显下跌：需检查市场环境、板块走弱、支撑失守及尾盘追高"
        elif change < 0:
            reason = "小幅回撤：观察支撑位和开盘承接"
        else:
            reason = "价格未下跌，记录正向样本"

        rows.append({
            "推荐日期": date, "代码": code, "名称": item.get("名称"),
            "前日参考价": old_price, "当前价": new_price,
            "T+1变化%": round(change, 2) if np.isfinite(change) else None,
            "复盘结论": reason,
            "前日评分": item.get("评分"),
            "前日风险提示": item.get("风险提示")
        })

    return date, pd.DataFrame(rows)


def save_learning(review):
    if review.empty:
        return

    path = os.path.join(REPORT_DIR, "overnight_learning.csv")
    if os.path.exists(path):
        try:
            old = pd.read_csv(path, dtype={"代码": str})
            review = pd.concat([old, review], ignore_index=True)
        except Exception:
            pass

    review = review.drop_duplicates(["推荐日期", "代码"], keep="last")
    review.to_csv(path, index=False, encoding="utf-8-sig")


def write_report(day, stats, picks, candidates, review, source_status):
    lines = [
        "# A股猎手｜隔夜超短报告",
        f"**日期：{day}　运行时间：{now_cn():%H:%M}**",
        "",
        "## 01｜市场温度",
        f"**{stats['市场热度']}**　上涨 {stats['上涨']} 家 / 下跌 {stats['下跌']} 家 / 平盘 {stats['平盘']} 家",
        f"涨停附近 {stats['涨停附近']} 家　跌停附近 {stats['跌停附近']} 家　上涨比例 {stats['上涨比例%']}%",
        f"两市成交额估算：{stats['成交额亿元']} 亿元",
        "",
        "## 02｜数据质量",
        f"股票列表：{source_status.get('universe_count')} 只；实时行情：{source_status.get('snapshot_count')} 只",
        f"腾讯行情：{source_status.get('tencent_count')} 只；新浪补充：{source_status.get('sina_count')} 只",
        "行业资金净流入、概念资金流及政策新闻：当前数据接口未接入，报告不虚构相关结论。",
        "",
        "## 03｜隔夜综合排名前三",
        "买入参考：14:50附近；次日卖出观察：9:30—9:40。止盈止损是参考价，不保证成交。"
    ]

    if picks.empty:
        lines.append("本轮没有取得有效技术分析结果。")
    else:
        for i, (_, x) in enumerate(picks.iterrows(), 1):
            lines += [
                "",
                f"### {i}. {x['名称']}（{x['代码']}）　**{x['评分']}/10分**",
                f"- 参考买入价：**{x['最新价']}元**",
                f"- 止盈：**{x['最低止盈价+0.5%']} / {x['止盈价+1%']} / {x['止盈价+2%']}元**",
                f"- 参考止损：**{x['参考止损价']}元**",
                f"- 支撑：{x['支撑位']}元　压力：{x['压力位']}元",
                f"- 技术理由：{x['评分理由']}",
                f"- 风险：{x['风险提示']}",
                f"- 分时数据：{x['数据完整性']}；板块资金：{x['板块资金']}"
            ]

    lines += ["", "## 04｜候选名单", f"本轮有效分析候选：{len(candidates)}只"]

    if not candidates.empty:
        lines.append("|排名|代码|名称|评分|涨幅|止盈+0.5%|止损|")
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        for i, (_, x) in enumerate(candidates.head(30).iterrows(), 1):
            lines.append(
                f"|{i}|{x['代码']}|{x['名称']}|{x['评分']}|{x['当日涨幅%']}%|"
                f"{x['最低止盈价+0.5%']}|{x['参考止损价']}|"
            )

    lines += ["", "## 05｜上一轮复盘"]

    if review.empty:
        lines.append("暂无可复盘的上一轮推荐。")
    else:
        for _, x in review.iterrows():
            lines.append(
                f"- {x['名称']}（{x['代码']}）：T+1变化 {x['T+1变化%']}%；"
                f"{x['复盘结论']}。原风险：{x.get('前日风险提示', '')}"
            )

    lines += [
        "", "## 06｜交易纪律",
        "- 0.5%/1%/2%为毛收益目标，需扣除佣金、印花税和滑点。",
        "- 次日若高开冲高，可按计划分批兑现；若低开跌破支撑，不应机械等待止盈。",
        "- A股T+1，买入当日不能卖出；跳空时止损价可能无法成交。",
        "- 本报告是量化筛选参考，不保证盈利，也不构成投资承诺。"
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
        price = fnum(s.get("price"))
        pct = fnum(s.get("change_pct"))

        if not np.isfinite(price) or not np.isfinite(pct):
            continue
        if price <= 0 or price > float(CONFIG["screening"]["max_price"]):
            continue
        if pct <= -10:
            continue
        if pct >= limit_rate(code, name) - 0.2:
            continue

        screen.append(s)

    log(f"硬性过滤后：{len(screen)}只，开始技术分析。")

    workers = int(CONFIG.get("data_source", {}).get("max_workers", 8))
    analyzed, errors = [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(analyze_one, s, CONFIG): s
            for s in screen
        }

        for i, future in enumerate(as_completed(futures), 1):
            try:
                item = future.result()
                if item:
                    analyzed.append(item)
            except Exception as exc:
                errors.append({
                    "代码": futures[future]["code"],
                    "名称": futures[future]["name"],
                    "错误": str(exc)[:300]
                })

            if i % 200 == 0 or i == len(futures):
                log(f"进度 {i}/{len(futures)}，有效分析 {len(analyzed)}")

    df = pd.DataFrame(analyzed)

    if not df.empty:
        df = df.sort_values(
            ["评分", "当日涨幅%"],
            ascending=[False, False]
        ).reset_index(drop=True)

    picks = df.head(3).copy() if not df.empty else pd.DataFrame()

    if not df.empty:
        df.to_csv(
            os.path.join(REPORT_DIR, f"candidates_{day}.csv"),
            index=False, encoding="utf-8-sig"
        )

    if not picks.empty:
        picks.to_csv(
            os.path.join(REPORT_DIR, f"picks_{day}.csv"),
            index=False, encoding="utf-8-sig"
        )

    prev_date, review = review_previous(snapshot)
    save_learning(review)

    if errors:
        pd.DataFrame(errors).to_csv(
            os.path.join(REPORT_DIR, "scan_errors.csv"),
            mode="a",
            header=not os.path.exists(os.path.join(REPORT_DIR, "scan_errors.csv")),
            index=False,
            encoding="utf-8-sig"
        )

    source_status = get_data_source_status()
    write_report(day, stats, picks, df, review, source_status)

    status = {
        "date": day,
        "market": stats,
        "candidate_count": len(df),
        "pick_count": len(picks),
        "errors": len(errors),
        "seconds": round(time.time() - start, 1),
        "data_source": source_status
    }

    with open(os.path.join(REPORT_DIR, "status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

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
