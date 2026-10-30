#!/usr/bin/env python3
"""14:30 尾盘筛选与下一交易日早盘复盘。仅生成研究信号，不连接券商。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import tempfile
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")
VERSION = "anytime-v2"
DEFAULTS = {
    "workers": 4, "request_timeout_seconds": 35,
    "history_days": 220, "intraday_max_age_seconds": 300,
    "shortlist_size": 60, "refresh_size": 8, "top_n": 3, "min_score": 6.0,
    "min_amount_yuan": 100000000, "min_price": 3, "max_price": 150,
    "min_day_pct": -1, "max_day_pct": 5.5, "max_pullback_pct": 2.2,
    "max_minute_age_seconds": 300, "min_minute_bars": 10,
    "allow_chinext": False, "min_market_breadth": 0.35,
    "target_net_pct": 0.0, "stop_net_pct": -1.5, "exit_time": "10:30:00",
    "round_trip_cost_pct": 0.20, "entry_slippage_pct": 0.05,
    "exit_slippage_pct": 0.05, "learning_window_sessions": 60,
    "min_learning_samples": 30, "min_learning_dates": 10,
    "min_learning_group": 12, "max_learning_penalty": 1.0,
    "apply_learning": True,
    "calendar_file": "",
}


def now():
    return datetime.now(TZ)


def at(day, text):
    return datetime.fromisoformat(day + "T" + text).replace(tzinfo=TZ)


def number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def code_of(value):
    value = str(value).removesuffix(".0")
    digits = "".join(x for x in value if x.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def stamp(text):
    value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return value.astimezone(TZ) if value.tzinfo else value.replace(tzinfo=TZ)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_json(path, default):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def public(row):
    return {k: v for k, v in row.items() if not k.startswith("_")}


def save_csv(path, rows, empty_fields=("code", "name", "score")):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [public(x) for x in rows]
    keys = list(dict.fromkeys(k for r in rows for k in r)) or list(empty_fields)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def config_id(cfg):
    # Only changes that alter labels / features invalidate learning samples.
    keys = ("target_net_pct", "stop_net_pct", "exit_time", "round_trip_cost_pct",
            "entry_slippage_pct", "exit_slippage_pct")
    packed = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    return VERSION + "-" + hashlib.sha256(packed.encode()).hexdigest()[:12]


def net_pct(entry, exit_price, cfg):
    """entry already includes entry slippage; no double deduction."""
    sell = exit_price * (1 - cfg["exit_slippage_pct"] / 100)
    return (sell / entry - 1) * 100 - cfg["round_trip_cost_pct"]


def clean_bars(rows, day, cutoff):
    result = {}
    for raw in rows:
        text = raw.get("时间", raw.get("day", ""))
        try:
            ts = stamp(text)
        except (TypeError, ValueError):
            continue
        clock = ts.strftime("%H:%M:%S")
        if ts.date().isoformat() != day or ts > cutoff:
            continue
        if not ("09:30:00" <= clock <= "11:30:00" or "13:00:00" <= clock <= "14:57:00"):
            continue
        close = number(raw.get("收盘", raw.get("close")))
        volume = number(raw.get("成交量", raw.get("volume")))
        if close is None or close <= 0 or volume is None or volume <= 0:
            continue
        result[ts] = {
            "ts": ts, "close": close,
            "high": number(raw.get("最高", raw.get("high")), close),
            "low": number(raw.get("最低", raw.get("low")), close),
            "volume": volume, "amount": number(raw.get("成交额", raw.get("amount")), 0),
            "vwap": number(raw.get("均价")),
        }
    return sorted(result.values(), key=lambda r: r["ts"])


def feature_tags(row):
    tags = []
    if row["day_pct"] >= 4: tags.append("当日涨幅>=4%")
    if row["deviation_ma10_pct"] >= 4: tags.append("偏离MA10>=4%")
    if row["turnover_pct"] >= 12: tags.append("换手率>=12%")
    if row["momentum15_pct"] < 0: tags.append("尾盘15分钟走弱")
    if row["above_vwap_pct"] >= 2: tags.append("偏离分时均价>=2%")
    if row["breadth"] < 0.45: tags.append("市场上涨占比<45%")
    return tags


def analyse(snapshot, rows, history, asof, cfg, breadth):
    """Only completed minute bars and prior-session daily bars enter features."""
    day = asof.date().isoformat()
    cutoff = asof.replace(second=0, microsecond=0) - timedelta(minutes=1)
    bars = clean_bars(rows, day, cutoff)
    if len(bars) < cfg["min_minute_bars"]:
        raise ValueError("当日有效分钟线不足，停牌/缺失数据不补值")
    latest = bars[-1]
    if (asof - latest["ts"]).total_seconds() > cfg["max_minute_age_seconds"]:
        raise ValueError("分钟行情过期")
    if bars[0]["ts"] > at(day, "09:35:00"):
        raise ValueError("缺少早盘分钟线，不能计算日内均价")
    prior = sorted([r for r in history if str(r.get("date", r.get("日期", "")))[:10] < day],
                   key=lambda r: str(r.get("date", r.get("日期", ""))))
    closes = [number(r.get("close", r.get("收盘"))) for r in prior]
    closes = [x for x in closes if x is not None and x > 0]
    if len(closes) < 40:
        raise ValueError("上市历史/历史日线不足40根")
    previous_date = str(prior[-1].get("date", prior[-1].get("日期", "")))[:10]
    if previous_date != cfg["_previous_session"]:
        raise ValueError("历史日线未更新到上一交易日")
    prev_close = number(snapshot.get("昨收"))
    if not prev_close or abs(closes[-1] / prev_close - 1) > 0.01:
        raise ValueError("日线/实时昨收不一致，可能除权或行情过期")
    price = latest["close"]
    amount = sum(x["amount"] for x in bars)
    vwap = latest["vwap"]
    if not vwap or vwap <= 0:
        vwap = amount / (sum(x["volume"] for x in bars) * 100)
    if not vwap or not (0.5 * price < vwap < 1.5 * price):
        raise ValueError("分时均价/成交量单位异常")
    past15 = [b for b in bars if b["ts"] <= latest["ts"] - timedelta(minutes=15)]
    if not past15:
        raise ValueError("缺少15分钟前价格")
    momentum = (price / past15[-1]["close"] - 1) * 100
    day_pct = (price / prev_close - 1) * 100
    high = max(b["high"] for b in bars)
    pullback = (1 - price / high) * 100
    ma10 = statistics.mean((closes + [price])[-10:])
    ma5 = statistics.mean((closes + [price])[-5:])
    above = (price / vwap - 1) * 100
    deviation = (price / ma10 - 1) * 100
    turnover = number(snapshot.get("换手率"))
    if turnover is None:
        raise ValueError("缺少换手率")
    if not (cfg["min_price"] <= price <= cfg["max_price"]): return None
    if not (cfg["min_day_pct"] <= day_pct <= cfg["max_day_pct"]): return None
    if amount < cfg["min_amount_yuan"] or pullback > cfg["max_pullback_pct"]: return None
    if not (-0.2 <= above <= 3 and -0.4 <= momentum <= 2.5 and 0.5 <= turnover <= 20): return None
    score, reasons = 0.0, []
    if 0 <= above <= 1.5: score += 2; reasons.append("价格贴近且高于分时均价")
    else: score += 0.5
    if 0 <= momentum <= 1.5: score += 2; reasons.append("尾盘温和走强")
    elif momentum >= 0: score += 1
    if pullback <= 1: score += 1
    if price >= ma10: score += 1
    if price >= ma5: score += 1
    if 0 <= day_pct <= 4: score += 1
    if amount >= cfg["min_amount_yuan"] * 2: score += 1
    if 2 <= turnover <= 12: score += 1
    row = {
        "signal_date": day, "code": code_of(snapshot["代码"]), "name": snapshot["名称"],
        "signal_time": latest["ts"].isoformat(), "received_at": asof.isoformat(),
        "reference_price": round(price, 4),
        "entry_proxy_price": round(price * (1 + cfg["entry_slippage_pct"] / 100), 6),
        "score_raw": round(score, 3), "score": round(score, 3),
        "day_pct": day_pct, "momentum15_pct": momentum, "pullback_pct": pullback,
        "above_vwap_pct": above, "deviation_ma10_pct": deviation,
        "turnover_pct": turnover, "amount_yuan": amount, "breadth": breadth,
        "reasons": reasons, "strategy_id": config_id(cfg), "learning_penalty": 0.0,
        "valid_until": at(day, "14:53:00").isoformat(), "_history": prior,
    }
    row["tags"] = feature_tags(row)
    return row


def review_signal(signal, rows, session, cfg):
    """Earliest completed-minute close trigger; never use the day's highest price as a sale."""
    if session <= signal["signal_date"]:
        raise ValueError("T+1复盘日期必须晚于入选日期")
    end = at(session, cfg["exit_time"])
    bars = clean_bars(rows, session, end)
    required_times = {at(session, "09:30:00") + timedelta(minutes=i) for i in range(1, int((end - at(session, "09:30:00")).total_seconds() // 60) + 1)}
    present = {b["ts"] for b in bars}
    if not required_times.issubset(present):
        raise ValueError("早盘分钟线不完整/停牌，结果标记待核验，不计失败")
    entry = signal["entry_proxy_price"]
    exit_bar, reason = bars[-1], "时间退出（研究默认）"
    peak = max(net_pct(entry, b["close"], cfg) for b in bars)
    for bar in bars:
        if bar["ts"] <= at(session, "09:30:00"): continue
        change = net_pct(entry, bar["close"], cfg)
        if change <= cfg["stop_net_pct"]:
            exit_bar, reason = bar, "止损触发（分钟收盘代理）"
            break
        if change > cfg["target_net_pct"]:
            exit_bar, reason = bar, "正净溢价触发（分钟收盘代理）"
            break
    # 一字/平盘分钟条可能没有可成交卖盘，不能据此宣称止损成交。
    ambiguous = exit_bar["high"] == exit_bar["low"]
    result = dict(public(signal))
    result.update({
        "review_session": session, "source": "minute_proxy",
        "exit_time": exit_bar["ts"].isoformat(), "exit_reference": exit_bar["close"],
        "net_return_pct": net_pct(entry, exit_bar["close"], cfg),
        "peak_close_net_pct": peak, "exit_reason": reason,
        "status": "execution_unverified" if ambiguous else "proxy_complete",
        "success": None if ambiguous else net_pct(entry, exit_bar["close"], cfg) > cfg["target_net_pct"],
        "execution_note": "分钟收盘价仅作代理；盘口、滑点、除权及真实成交需人工核验",
    })
    return result


def learn(outcomes, cfg, before_session):
    """Use success AND failure samples from earlier observations, grouped by date."""
    samples = [r for r in outcomes if r.get("status") == "proxy_complete"
               and r.get("source") == "minute_proxy" and r.get("strategy_id") == config_id(cfg)
               and r.get("review_session", "9999") <= before_session]
    dates = sorted(set(r["signal_date"] for r in samples))[-cfg["learning_window_sessions"]:]
    samples = [r for r in samples if r["signal_date"] in dates]
    result = {"strategy_id": config_id(cfg), "samples": len(samples), "dates": len(dates), "rules": [],
              "apply_learning": cfg["apply_learning"], "note": "相关特征统计，不是下跌因果；达到样本门槛后最多扣1分，尚未证实样本外改善"}
    if len(samples) < cfg["min_learning_samples"] or len(dates) < cfg["min_learning_dates"]:
        result["note"] += "；样本不足，不生成扣分规则"
        return result
    for tag in sorted({tag for r in samples for tag in r.get("tags", [])}):
        group = [r for r in samples if tag in r.get("tags", [])]
        other = [r for r in samples if tag not in r.get("tags", [])]
        if min(len(group), len(other)) < cfg["min_learning_group"]: continue
        if min(len({r["signal_date"] for r in group}), len({r["signal_date"] for r in other})) < 5: continue
        # One busy trading day must not dominate the comparison.
        def by_day(rows, fn):
            return statistics.mean(statistics.mean(fn(r) for r in rows if r["signal_date"] == d)
                                   for d in set(r["signal_date"] for r in rows))
        loss = by_day(group, lambda r: float(not r["success"]))
        control = by_day(other, lambda r: float(not r["success"]))
        avg = by_day(group, lambda r: r["net_return_pct"])
        excess = loss - control
        penalty = min(0.5, max(0, excess)) if excess >= 0.15 and avg < 0 else 0.0
        result["rules"].append({"tag": tag, "n": len(group), "control_n": len(other),
                                "failure_rate": loss, "control_failure_rate": control,
                                "mean_net_pct": avg, "penalty": penalty})
    return result


def apply_learning(rows, rules, cfg):
    for row in rows:
        proposed = min(cfg["max_learning_penalty"], sum(r["penalty"] for r in rules["rules"] if r["tag"] in row["tags"]))
        row["proposed_learning_penalty"] = proposed
        row["learning_penalty"] = proposed if cfg["apply_learning"] else 0
        row["score"] = row["score_raw"] - row["learning_penalty"]
    return sorted(rows, key=lambda r: (-r["score"], -r["amount_yuan"], r["code"]))



def daily_analyse(snapshot, history, asof_text, cfg, breadth):
    """Anytime fallback: score the latest completed daily bar, never pretend it is live intraday data."""
    asof = stamp(asof_text)
    rows = []
    for raw in history:
        day = str(raw.get("日期", raw.get("date", "")))[:10]
        close = number(raw.get("收盘", raw.get("close")))
        if len(day) == 10 and close and close > 0:
            rows.append({"date": day, "close": close,
                         "open": number(raw.get("开盘", raw.get("open")), close),
                         "high": number(raw.get("最高", raw.get("high")), close),
                         "low": number(raw.get("最低", raw.get("low")), close),
                         "amount": number(raw.get("成交额", raw.get("amount")), 0),
                         "pct": number(raw.get("涨跌幅", raw.get("pct")), 0),
                         "turnover": number(raw.get("换手率", raw.get("turnover")), 0)})
    rows.sort(key=lambda x: x["date"])
    if len(rows) < 35: raise ValueError("有效日线不足35根")
    live_session = ("09:30" <= asof.strftime("%H:%M") <= "11:30") or ("13:00" <= asof.strftime("%H:%M") <= "15:00")
    if rows[-1]["date"] == asof.date().isoformat():
        rows = rows[:-1]  # today is incomplete until the close auction has finished
    if not rows: raise ValueError("尚无已完成日线")
    last = rows[-1]
    closes = [r["close"] for r in rows]
    prev = closes[-2] if len(closes) > 1 else closes[-1]
    price = number(snapshot.get("最新价"), last["close"]) if live_session else last["close"]
    if not price or price <= 0: price = last["close"]
    ema12 = ema26 = closes[0]
    for value in closes[1:]:
        ema12 = value * (2/13) + ema12 * (11/13)
        ema26 = value * (2/27) + ema26 * (25/27)
    macd = ema12 - ema26
    signal_line = macd
    # Calculate MACD signal from rolling prefixes to avoid a one-point approximation.
    fast = slow = closes[0]; macds=[]
    for value in closes[1:]:
        fast = value * (2/13) + fast * (11/13)
        slow = value * (2/27) + slow * (25/27)
        macds.append(fast-slow)
    sig = macds[0] if macds else 0
    for value in macds[1:]: sig = value*(2/10)+sig*(8/10)
    dif = macds[-1] if macds else 0
    changes = [closes[i]-closes[i-1] for i in range(max(1,len(closes)-14),len(closes))]
    gains = [max(x,0) for x in changes]; losses=[max(-x,0) for x in changes]
    avg_gain = sum(gains)/max(1,len(gains)); avg_loss=sum(losses)/max(1,len(losses))
    rsi = 100 if avg_loss == 0 else 100 - 100/(1+avg_gain/avg_loss)
    prev = number(snapshot.get("昨收"), prev) if live_session else prev
    pct = (price/prev-1)*100 if prev else 0
    ma5 = statistics.mean(closes[-5:]); ma10=statistics.mean(closes[-10:]); ma60=statistics.mean(closes[-60:]) if len(closes)>=60 else statistics.mean(closes)
    amount = number(snapshot.get("成交额"), last["amount"]) if live_session else last["amount"]
    turnover = (number(snapshot.get("换手率"),0) if live_session else last["turnover"]) or number(snapshot.get("换手率"),0)
    score=0.0; reasons=[]; tags=[]
    if dif >= sig and dif > 0: score+=2; reasons.append("近10日MACD偏强")
    if 40 <= rsi <= 68: score+=1.5; reasons.append("RSI处于相对温和区间")
    elif rsi < 40: score+=1; reasons.append("RSI偏低，需确认止跌")
    if price >= ma5: score+=1; reasons.append("收盘站上MA5")
    if price >= ma10: score+=1; reasons.append("收盘站上MA10")
    if price >= ma60: score+=1
    if -1 <= pct <= 4: score+=1; reasons.append("当日涨幅未明显过热")
    if amount >= cfg["min_amount_yuan"]: score+=1; reasons.append("成交额达到门槛")
    if 1 <= turnover <= 12: score+=1
    if pct >= 5: tags.append("当日涨幅偏高")
    if rsi >= 75: tags.append("RSI过热")
    if price < ma10: tags.append("收盘低于MA10")
    if breadth < .45: tags.append("市场上涨占比<45%")
    if not cfg["min_price"] <= price <= cfg["max_price"]: return None
    if amount < cfg["min_amount_yuan"] or score < 3: return None
    code=code_of(snapshot.get("代码",snapshot.get("code","")))
    signal_day = asof.date().isoformat() if live_session else last["date"]
    return {"signal_date":signal_day,"code":code,"name":snapshot.get("名称",snapshot.get("name",code)),
            "signal_time":asof.isoformat() if live_session else last["date"]+"T15:00:00+08:00","received_at":asof.isoformat(),
            "reference_price":round(price,4),"entry_proxy_price":round(price,4),
            "score_raw":round(min(10,score),3),"score":round(min(10,score),3),
            "day_pct":pct,"momentum15_pct":None,"pullback_pct":(1-price/max(r["high"] for r in rows[-10:]))*100,
            "above_vwap_pct":None,"deviation_ma10_pct":(price/ma10-1)*100,"turnover_pct":turnover,
            "amount_yuan":amount,"breadth":breadth,"rsi14":round(rsi,2),"macd_dif":round(dif,4),
            "macd_signal":round(sig,4),"ma5":round(ma5,4),"ma10":round(ma10,4),"ma60":round(ma60,4),
            "reasons":reasons,"tags":tags,"strategy_id":config_id(cfg),"learning_penalty":0.0,
            "data_mode":"live_snapshot_plus_completed_daily" if live_session else "latest_completed_daily",
            "data_date":last["date"],"valid_until":"下一次运行前需重新核验"}

def _worker(task, destination):
    """Network work runs in killable child processes; use documented AkShare APIs only."""
    try:
        if task["kind"] == "test_hang":
            time.sleep(task["seconds"]); data = True
        else:
            import akshare as ak
            import pandas as pd

            def records(frame):
                if frame is None or getattr(frame, "empty", False): return []
                return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))

            def first_available(names):
                for name in names:
                    fn = getattr(ak, name, None)
                    if callable(fn): return fn
                raise AttributeError("AkShare缺少可用接口：" + "/".join(names))

            kind = task["kind"]
            if kind == "calendar":
                fn = first_available(["tool_trade_date_hist_sina"])
                frame = fn()
                data = sorted(str(d)[:10] for d in frame["trade_date"])
            elif kind == "spot":
                frame = None; used = None; failures = []
                for name in ("stock_zh_a_spot_em", "stock_zh_a_spot"):
                    fn = getattr(ak, name, None)
                    if not callable(fn): continue
                    try:
                        frame = fn(); used = name
                        if frame is not None and not getattr(frame, "empty", False): break
                        failures.append(name + ": 返回空数据")
                    except Exception as exc:
                        failures.append(name + ": " + type(exc).__name__ + ": " + str(exc)[:120])
                if frame is None or getattr(frame, "empty", False):
                    raise RuntimeError("全市场快照接口均失败：" + "；".join(failures))
                data = {"rows": records(frame), "received_at": now().isoformat(), "api": used, "fallback_errors": failures}
            elif kind in ("daily", "analyse", "refresh"):
                code = task["code"]
                end_day = task.get("day", now().date().isoformat())
                start = (datetime.fromisoformat(end_day) - timedelta(days=task.get("history_days", 220))).strftime("%Y%m%d")
                end = end_day.replace("-", "")
                history = []; failures = []
                fn = getattr(ak, "stock_zh_a_hist", None)
                if callable(fn):
                    try:
                        history = records(fn(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq"))
                    except Exception as exc:
                        failures.append("stock_zh_a_hist: " + type(exc).__name__ + ": " + str(exc)[:120])
                if not history:
                    tx = getattr(ak, "stock_zh_a_hist_tx", None)
                    if callable(tx):
                        try:
                            prefix = ("sh" if code.startswith(("5", "6", "9")) else "sz") + code
                            history = records(tx(symbol=prefix, start_date=start, end_date=end, adjust="qfq"))
                        except Exception as exc:
                            failures.append("stock_zh_a_hist_tx: " + type(exc).__name__ + ": " + str(exc)[:120])
                if not history:
                    raise RuntimeError("日线接口均失败：" + "；".join(failures or ["AkShare没有可用日线接口"]))
                if kind == "daily":
                    data = history
                else:
                    data = daily_analyse(task["snapshot"], history, task["asof"], task["cfg"], task["breadth"])
            elif kind == "minutes":
                code, day = task["code"], task["day"]
                fn = getattr(ak, "stock_zh_a_hist_min_em", None)
                if callable(fn):
                    frame = fn(symbol=code, period="1", adjust="", start_date=day + " 09:30:00", end_date=day + " " + task.get("end_time", "15:00:00"))
                else:
                    fn = first_available(["stock_zh_a_minute"])
                    prefix = ("sh" if code.startswith(("5", "6", "9")) else "sz") + code
                    frame = fn(symbol=prefix, period="1")
                data = records(frame)
                if task.get("archive_path"):
                    save_json(task["archive_path"], {"minutes": data, "fetched_at": now().isoformat(), "code": code, "day": day})
            else:
                raise ValueError("Unknown task kind: " + str(kind))
        save_json(destination, {"ok": True, "data": data})
    except Exception as exc:
        save_json(destination, {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:400]})


def bounded_jobs(tasks, deadline, cfg):
    """Bound each call AND the entire batch; no ThreadPool shutdown waiting on blocked requests."""
    ctx = mp.get_context("spawn")
    results, errors, running = {}, [], {}
    queue = list(enumerate(tasks))
    with tempfile.TemporaryDirectory(prefix="ashare_jobs_") as temp:
        try:
            while (queue or running) and now() < deadline:
                while queue and len(running) < cfg["workers"] and now() < deadline:
                    index, task = queue.pop(0)
                    path = Path(temp) / (str(index) + ".json")
                    process = ctx.Process(target=_worker, args=(task, str(path)))
                    process.start()
                    running[index] = (process, time.monotonic(), task, path)
                for index, (process, started, task, path) in list(running.items()):
                    expired = time.monotonic() - started >= cfg["request_timeout_seconds"]
                    if process.is_alive() and not expired: continue
                    if process.is_alive(): process.terminate()
                    process.join(timeout=0.1)
                    if process.is_alive(): process.kill(); process.join(timeout=0.1)
                    response = load_json(path, {"ok": False, "error": "请求超时/子进程退出"})
                    if expired: response = {"ok": False, "error": "请求超时"}
                    if response["ok"]: results[task["id"]] = response["data"]
                    else: errors.append({"task": task["id"], "error": response["error"]})
                    del running[index]
                time.sleep(0.03)
        finally:
            for process, _, task, _ in running.values():
                process.terminate(); process.join(timeout=0.1)
                if process.is_alive(): process.kill(); process.join(timeout=0.1)
                errors.append({"task": task["id"], "error": "阶段截止，任务已终止"})
            errors.extend({"task": task["id"], "error": "阶段截止，未执行"} for _, task in queue)
    return results, errors


def wait_until(target):
    while now() < target:
        time.sleep(min(10, max(0.01, (target - now()).total_seconds())))


def shortlist(rows, cfg):
    allowed = ("000", "001", "002", "003", "600", "601", "603", "605", "688", "689", "300", "301")
    result = []
    for row in rows:
        name, code = str(row.get("名称", "")), code_of(row.get("代码", ""))
        if not code.startswith(allowed) or any(x in name.upper() for x in ("ST", "退", "N", "C")): continue
        price, pct, amount = (number(row.get(x)) for x in ("最新价", "涨跌幅", "成交额"))
        if any(x is None for x in (price, pct, amount)): continue
        if not cfg["min_price"] <= price <= cfg["max_price"]: continue
        if not cfg["min_day_pct"] <= pct <= cfg["max_day_pct"]: continue
        if amount < cfg["min_amount_yuan"]: continue
        row = dict(row, 代码=code)
        result.append(row)
    # Heuristic preselection only, not an exhaustive market scan.
    def priority(r):
        return (-(abs(number(r.get("涨跌幅"), 0) - 2)), number(r.get("成交额"), 0))
    return sorted(result, key=priority, reverse=True)[:cfg["shortlist_size"]]


def review_previous(root, today, calendar, cfg, deadline):
    previous = max(d for d in calendar if d < today)
    signals = load_json(root / "signals" / (previous + ".json"), [])
    pending = []
    for signal in signals:
        if signal.get("strategy_id") == config_id(cfg):
            pending.append({"id": signal["code"], "kind": "minutes", "code": signal["code"],
                            "day": today, "end_time": cfg["exit_time"],
                            "archive_path": str(root / "raw" / today / (signal["code"] + "_review.json"))})
    raw, errors = bounded_jobs(pending, deadline, cfg)
    history_path = root / "learning_outcomes.json"
    outcomes = load_json(history_path, [])
    reviewed = []
    for signal in signals:
        if signal.get("strategy_id") != config_id(cfg):
            errors.append({"task": signal["code"], "error": "退出规则/成本已改变，跳过旧策略样本"}); continue
        if signal["code"] not in raw: continue
        try:
            result = review_signal(signal, raw[signal["code"]], today, cfg)
        except ValueError as exc:
            result = dict(public(signal), review_session=today, status="pending_data", source="minute_proxy", success=None, error=str(exc))
        reviewed.append(result)
    # Idempotent: reruns on the same session do not create extra observations.
    unique = {(r["signal_date"], r["code"], r.get("strategy_id")): r for r in outcomes}
    unique.update({(r["signal_date"], r["code"], r.get("strategy_id")): r for r in reviewed})
    outcomes = list(unique.values())
    save_json(history_path, outcomes)
    save_csv(root / "review" / (today + ".csv"), reviewed)
    save_csv(root / "learning_outcomes.csv", outcomes)
    rules = learn(outcomes, cfg, today)
    save_json(root / "learned_rules.json", rules)
    return rules, errors, previous, len(signals)


def report(root, day, picks, status, cfg):
    folder = root / "daily" / day
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "status.json", status)
    save_csv(folder / "picks.csv", picks)
    lines = ["# 尾盘隔夜研究报告 " + day, "", "状态：" + status["state"],
             "生成时间：" + now().isoformat(), "", "14:30开始；14:47刷新快照；14:48复核分钟线；14:49:10停止网络任务；争取14:50前写出结果。",
             "退出口径：下一交易日已完成的分钟收盘价，扣除估算成本后净收益 > " + str(cfg["target_net_pct"]) + "% 即触发退出代理。",
             "研究默认：净止损 " + str(cfg["stop_net_pct"]) + "%，最晚 " + cfg["exit_time"] + " 退出；这两项未经用户确认，不是实盘指令。",
             "费用参数是估计值，须按券商收费、最低佣金和交易金额修改；未接入券商，不自动下单或监控卖出。",
             "", "## 候选（分数不代表胜率）", ""]
    for i, row in enumerate(picks, 1):
        lines += [f"{i}. {row['name']} {row['code']}｜{row['score']:.2f}/10｜分钟参考价 {row['reference_price']}",
                  f"   行情时间 {row['signal_time']}；数据获取时间 {row['received_at']}；有效至 {row['valid_until']}",
                  "   理由：" + "；".join(row["reasons"]), "   复盘特征：" + "；".join(row["tags"]), ""]
    if not picks: lines += ["无可用候选。请看状态和错误明细；接口故障或样本覆盖不足不等于市场没有机会。", ""]
    lines += ["## 数据与学习", "", json.dumps(status, ensure_ascii=False, indent=2), "",
              "raw文件夹留存原始分钟线；缺失数据标记待核验，不算失败或成功。",
              "候选价是研究买入代理，不是真实买入；分钟收盘触发不是实际可成交证明。",
              "复盘同时保留成功与失败样本；只读取已完成的下一交易日早盘，周末和节假日按交易日历处理。",
              "达到样本数和跨日期门槛后，学习最多扣1分；可设apply_learning=false先做影子观察。尚未证实它会提高收益。",
              "停牌、封死跌停、除权除息、公告事件及无法成交需要人工核验。"]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("报告：", folder / "report.md", flush=True)


def validate_config(cfg):
    if not 1 <= cfg["workers"] <= 8: raise ValueError("并发数须1—8")
    if not 1 <= cfg["top_n"] <= 10: raise ValueError("候选数量设置无效")
    if cfg["min_price"] <= 0 or cfg["max_price"] <= cfg["min_price"]: raise ValueError("价格区间设置无效")


def scan_market(cfg=None, root=None, wait=False):
    """Manual anytime scan: live snapshot during session; latest completed daily bars outside session."""
    cfg = dict(DEFAULTS, **(cfg or {})); validate_config(cfg)
    root = Path(root or Path(__file__).resolve().parent / "reports")
    root.mkdir(parents=True, exist_ok=True)
    started = now(); day = started.date().isoformat()
    folder = root / "daily" / (day + "_" + started.strftime("%H%M%S"))
    status = {"state":"正在获取行情", "version":VERSION, "started_at":started.isoformat(),
              "errors":[], "mode":"intraday" if (("09:30" <= started.strftime("%H:%M") <= "11:30") or ("13:00" <= started.strftime("%H:%M") <= "15:00")) else "daily"}
    picks=[]
    try:
        # Fetch a fresh all-market snapshot on every manual run.
        got, errors = bounded_jobs([{"id":"spot","kind":"spot"}], started+timedelta(seconds=cfg["request_timeout_seconds"]), cfg)
        status["errors"] += errors
        if "spot" not in got or not got["spot"].get("rows"):
            raise RuntimeError("全市场快照获取失败；本轮不输出虚假空选股结果")
        spot=got["spot"]["rows"]; status["snapshot_received_at"]=got["spot"]["received_at"]; status["snapshot_api"]=got["spot"].get("api")
        # Refresh the existing T+1 learning pipeline when today is a real trading session.
        cal, cal_errors = bounded_jobs([{"id":"calendar","kind":"calendar"}], now()+timedelta(seconds=25), cfg)
        status["errors"] += cal_errors
        calendar = cal.get("calendar", [])
        rules = load_json(root/"learned_rules.json", {"rules":[]})
        if day in calendar:
            try:
                rules, review_errors, prev_day, prev_count = review_previous(root, day, calendar, cfg, now()+timedelta(seconds=90))
                status["previous_signal_date"], status["previous_signals"] = prev_day, prev_count
                status["learning_samples"] = rules.get("samples", 0)
                status["errors"] += review_errors
            except Exception as review_exc:
                status["errors"].append({"task":"T+1复盘","error":type(review_exc).__name__+": "+str(review_exc)[:250]})
        else:
            status["learning_note"] = "非交易日或交易日历不可用，本次不执行T+1分钟复盘"
        valid=[r for r in spot if number(r.get("涨跌幅")) is not None and number(r.get("成交额"),0)>0]
        breadth=sum(number(r.get("涨跌幅"),0)>0 for r in valid)/max(1,len(valid))
        status["market_breadth"]=round(breadth,4)
        candidates=shortlist(spot,cfg); status["shortlisted"]=len(candidates)
        if not candidates: raise RuntimeError("快照正常，但按价格/涨幅/成交额过滤后没有初选股")
        jobs=[{"id":r["代码"],"kind":"daily","code":r["代码"],"day":day,"history_days":cfg["history_days"]} for r in candidates]
        daily, errs=bounded_jobs(jobs, now()+timedelta(seconds=max(60,cfg["request_timeout_seconds"]*len(jobs)/max(1,cfg["workers"]))), cfg)
        status["errors"]+=errs; status["daily_completed"]=len(daily)
        scored=[]
        for snap in candidates:
            code=snap["代码"]
            if code not in daily: continue
            try:
                row=daily_analyse(snap,daily[code],now().isoformat(),cfg,breadth)
                if row: scored.append(row)
            except Exception as exc:
                status["errors"].append({"task":code,"error":type(exc).__name__+": "+str(exc)[:250]})
        status["qualified_before_learning"]=len(scored)
        # Keep historical learning files intact; apply existing learned rules if compatible.
        scored=apply_learning(scored,rules,cfg)
        picks=[public(r) for r in scored if r["score"]>=cfg["min_score"]][:cfg["top_n"]]
        status["state"]="研究候选已生成，需核对实时盘口" if picks else "数据已读取，当前无达到评分门槛的候选"
        status["data_mode"]="盘中最新快照价格+最近已完成日线指标" if status["mode"]=="intraday" else "最近已完成日线收盘+全市场快照；非盘中实时价"
        status["data_date"] = max((r.get("data_date","") for r in picks), default="")
        save_csv(folder/"picks.csv",picks)
        save_csv(folder/"candidates.csv",scored)
        if picks:
            signal_day = max((r.get("signal_date", day) for r in picks), default=day)
            save_json(root/"signals"/(signal_day+".json"),picks)
        save_json(folder/"status.json",status)
        lines=["# A股猎手研究报告", "", "运行时间："+started.isoformat(), "运行模式："+status["mode"],
               "数据说明："+status["data_mode"], "市场上涨占比："+str(round(breadth*100,2))+"%", "", "## 候选"]
        if not picks: lines += ["本轮没有达到评分门槛的股票。请区分市场无机会、过滤条件过严和单股数据缺失。"]
        for i,r in enumerate(picks,1):
            lines += [f"{i}. {r['name']}（{r['code']}）｜{r['score']:.2f}/10｜参考价 {r['reference_price']}",
                      "   数据日期："+str(r.get("data_date"))+"；RSI14："+str(r.get("rsi14"))+"；MACD DIF/DEA："+str(r.get("macd_dif"))+"/"+str(r.get("macd_signal")),
                      "   理由："+"；".join(r.get("reasons",[]))]
        lines += ["", "## 运行状态", json.dumps(status,ensure_ascii=False,indent=2),
                  "", "评分是研究排序，不是收益承诺；参考价不是实际成交价。盘后运行读取最近完成日线，不会伪装成实时盘中行情。"]
        (folder/"report.md").write_text("\n".join(lines),encoding="utf-8")
        print("报告：",folder/"report.md",flush=True)
        return picks
    except Exception as exc:
        status["state"]="运行失败，未生成有效选股结果"
        status["errors"].append({"error":type(exc).__name__+": "+str(exc)[:500]})
        folder.mkdir(parents=True,exist_ok=True)
        save_json(folder/"status.json",status); save_csv(folder/"picks.csv",[])
        (folder/"report.md").write_text("# A股猎手运行失败\n\n"+json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
        print("运行失败报告：",folder/"report.md",flush=True)
        # Do not let GitHub Actions report a green success when the scan itself failed.
        raise RuntimeError("A股猎手扫描失败；详情见 reports/daily/*/status.json 和 report.md") from exc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--output",default=str(Path(__file__).with_name("reports")))
    parser.add_argument("--wait",action="store_true",help="兼容旧命令；新版不等待固定时间，立即运行")
    args=parser.parse_args()
    cfg=dict(DEFAULTS,**load_json(args.config,{}))
    scan_market(cfg,args.output,False)


if __name__ == "__main__":
    mp.freeze_support(); main()
