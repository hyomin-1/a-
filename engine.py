#!/usr/bin/env python3
"""随时可运行的筛选与复盘。采用 efinance 接口。"""
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
VERSION = "manual-v7-ef"
DEFAULTS = {
    "workers": 2, "request_timeout_seconds": 40,
    "shortlist_size": 60, "refresh_size": 8, "top_n": 3, "min_score": 6.0,
    "min_amount_yuan": 100000000, "min_price": 3, "max_price": 150,
    "min_day_pct": -1, "max_day_pct": 5.5, "max_pullback_pct": 2.2,
    "min_minute_bars": 180,
    "allow_chinext": False, "min_market_breadth": 0.35,
    "target_net_pct": 0.0, "stop_net_pct": -1.5, "exit_time": "10:30:00",
    "round_trip_cost_pct": 0.20, "entry_slippage_pct": 0.05,
    "exit_slippage_pct": 0.05, "learning_window_sessions": 60,
    "min_learning_samples": 30, "min_learning_dates": 10,
    "min_learning_group": 12, "max_learning_penalty": 1.0,
    "apply_learning": True,
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
    keys = ("target_net_pct", "stop_net_pct", "exit_time", "round_trip_cost_pct",
            "entry_slippage_pct", "exit_slippage_pct")
    packed = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    return VERSION + "-" + hashlib.sha256(packed.encode()).hexdigest()[:12]

def net_pct(entry, exit_price, cfg):
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
    day = asof.date().isoformat()
    cutoff = asof.replace(second=0, microsecond=0) - timedelta(minutes=1)
    bars = clean_bars(rows, day, cutoff)
    if len(bars) < cfg["min_minute_bars"]:
        raise ValueError("当日有效分钟线不足")
    latest = bars[-1]
    
    if (asof - latest["ts"]).total_seconds() > 86400:
        raise ValueError("不是当天的分钟线")

    prior = [r for r in history if str(r.get("date", ""))[:10] < day]
    prior.sort(key=lambda r: str(r.get("date", "")))
    closes = [number(r.get("close")) for r in prior]
    closes = [x for x in closes if x is not None and x > 0]
    if len(closes) < 40:
        raise ValueError("上市历史日线不足40根")
        
    prev_close = number(snapshot.get("昨收"))
    if not prev_close or abs(closes[-1] / prev_close - 1) > 0.05:
        raise ValueError("日线/实时昨收不一致")
    price = latest["close"]
    amount = sum(x["amount"] for x in bars)
    vwap = latest["vwap"]
    if not vwap or vwap <= 0:
        vwap = amount / (sum(x["volume"] for x in bars) * 100)
    if not vwap or not (0.5 * price < vwap < 1.5 * price):
        raise ValueError("分时均价异常")
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
        "above_vwap_pct": above, "deviation_ma10_pct": deviation, "turnover_pct": turnover,
        "amount_yuan": amount, "breadth": breadth,
        "reasons": reasons, "strategy_id": config_id(cfg), "learning_penalty": 0.0,
        "valid_until": at(day, "23:59:59").isoformat(), "_history": prior,
    }
    row["tags"] = feature_tags(row)
    return row

def review_signal(signal, rows, session, cfg):
    if session <= signal["signal_date"]:
        raise ValueError("T+1复盘日期必须晚于入选日期")
    end = at(session, cfg["exit_time"])
    bars = clean_bars(rows, session, end)
    required_times = {at(session, "09:30:00") + timedelta(minutes=i) for i in range(1, int((end - at(session, "09:30:00")).total_seconds() // 60) + 1)}
    present = {b["ts"] for b in bars}
    if not required_times.issubset(present):
        raise ValueError("早盘分钟线不完整")
    entry = signal["entry_proxy_price"]
    exit_bar, reason = bars[-1], "时间退出"
    peak = max(net_pct(entry, b["close"], cfg) for b in bars)
    for bar in bars:
        if bar["ts"] <= at(session, "09:30:00"): continue
        change = net_pct(entry, bar["close"], cfg)
        if change <= cfg["stop_net_pct"]:
            exit_bar, reason = bar, "止损触发"
            break
        if change > cfg["target_net_pct"]:
            exit_bar, reason = bar, "正净溢价触发"
            break
    ambiguous = exit_bar["high"] == exit_bar["low"]
    result = dict(public(signal))
    result.update({
        "review_session": session, "source": "minute_proxy",
        "exit_time": exit_bar["ts"].isoformat(), "exit_reference": exit_bar["close"],
        "net_return_pct": net_pct(entry, exit_bar["close"], cfg),
        "peak_close_net_pct": peak, "exit_reason": reason,
        "status": "execution_unverified" if ambiguous else "proxy_complete",
        "success": None if ambiguous else net_pct(entry, exit_bar["close"], cfg) > cfg["target_net_pct"],
        "execution_note": "分钟收盘价仅作代理",
    })
    return result

def learn(outcomes, cfg, before_session):
    samples = [r for r in outcomes if r.get("status") == "proxy_complete"
               and r.get("source") == "minute_proxy" and r.get("strategy_id") == config_id(cfg)
               and r.get("review_session", "9999") <= before_session]
    dates = sorted(set(r["signal_date"] for r in samples))[-cfg["learning_window_sessions"]:]
    samples = [r for r in samples if r["signal_date"] in dates]
    result = {"strategy_id": config_id(cfg), "samples": len(samples), "dates": len(dates), "rules": [],
              "apply_learning": cfg["apply_learning"], "note": "相关特征统计，不是因果"}
    if len(samples) < cfg["min_learning_samples"] or len(dates) < cfg["min_learning_dates"]:
        return result
    for tag in sorted({tag for r in samples for tag in r.get("tags", [])}):
        group = [r for r in samples if tag in r.get("tags", [])]
        other = [r for r in samples if tag not in r.get("tags", [])]
        if min(len(group), len(other)) < cfg["min_learning_group"]: continue
        if min(len({r["signal_date"] for r in group}), len({r["signal_date"] for r in other})) < 5: continue
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

def _worker(task, destination):
    try:
        import efinance as ef
        import pandas as pd
        
        def records(df):
            if df is None or df.empty:
                return []
            # efinance 返回的列名是中文，直接转字典
            return df.to_dict(orient="records")

        kind = task["kind"]
        
        if kind == "spot":
            # 获取全市场快照 (efinance 有内置的重试和伪装)
            df = ef.stock.get_realtime_quotes()
            if df is None or df.empty:
                raise ValueError("未能获取到 efinance 实时行情")
            data = {"rows": records(df), "received_at": now().isoformat()}
            
        elif kind in ("analyse", "refresh", "minutes"):
            day, code = task["day"], task["code"]
            
            # efinance 分钟线
            df_min = ef.stock.get_quote_history(code, klt=1, fqt=0)
            if df_min is not None and not df_min.empty:
                df_min["时间"] = df_min["日期"]
                minutes = records(df_min)
            else:
                minutes = []
                
            if kind == "minutes":
                data = minutes
            else:
                history = task.get("history")
                if history is None:
                    # 日线
                    df_hist = ef.stock.get_quote_history(code, klt=101, fqt=1)
                    if df_hist is not None and not df_hist.empty:
                        df_hist["date"] = df_hist["日期"]
                        df_hist["close"] = df_hist["收盘"]
                        history = records(df_hist)
                    else:
                        history = []
                data = analyse(task["snapshot"], minutes, history, now(), task["cfg"], task["breadth"])
        else:
            raise ValueError("Unknown task kind")
        save_json(destination, {"ok": True, "data": data})
    except Exception as exc:
        import traceback
        save_json(destination, {"ok": False, "error": type(exc).__name__ + ": " + str(exc) + "\n" + traceback.format_exc()})

def bounded_jobs(tasks, deadline, cfg):
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
                    response = load_json(path, {"ok": False, "error": "请求超时/退出"})
                    if expired: response = {"ok": False, "error": "请求超时"}
                    if response["ok"]: results[task["id"]] = response["data"]
                    else: errors.append({"task": task["id"], "error": response["error"]})
                    del running[index]
                time.sleep(0.03)
        finally:
            for process, _, task, _ in running.values():
                process.terminate(); process.join(timeout=0.1)
                if process.is_alive(): process.kill(); process.join(timeout=0.1)
                errors.append({"task": task["id"], "error": "阶段截止"})
            errors.extend({"task": task["id"], "error": "未执行"} for _, task in queue)
    return results, errors

def shortlist(rows, cfg):
    allowed = ("000", "001", "002", "003", "600", "601", "603", "605")
    if cfg["allow_chinext"]: allowed += ("300", "301")
    result = []
    for row in rows:
        name, code = str(row.get("名称", "")), code_of(row.get("代码", ""))
        if not code.startswith(allowed) or any(x in name.upper() for x in ("ST", "退", "N", "C")): continue
        
        price = number(row.get("最新价"))
        pct = number(row.get("涨跌幅"))
        amount = number(row.get("成交额"))
        
        if any(x is None for x in (price, pct, amount)): continue
        if not cfg["min_price"] <= price <= cfg["max_price"]: continue
        if not cfg["min_day_pct"] <= pct <= cfg["max_day_pct"]: continue
        if amount < cfg["min_amount_yuan"]: continue
        row = dict(row, 代码=code)
        result.append(row)
    def priority(r):
        return (-(abs(number(r.get("涨跌幅"), 0) - 2)), number(r.get("成交额"), 0))
    return sorted(result, key=priority, reverse=True)[:cfg["shortlist_size"]]

def review_previous(root, today, previous, cfg, deadline):
    signals = load_json(root / "signals" / (previous + ".json"), [])
    pending = []
    for signal in signals:
        if signal.get("strategy_id") == config_id(cfg):
            pending.append({"id": signal["code"], "kind": "minutes", "code": signal["code"],
                            "day": today, "end_time": cfg["exit_time"]})
    raw, errors = bounded_jobs(pending, deadline, cfg)
    history_path = root / "learning_outcomes.json"
    outcomes = load_json(history_path, [])
    reviewed = []
    for signal in signals:
        if signal.get("strategy_id") != config_id(cfg): continue
        if signal["code"] not in raw: continue
        try:
            result = review_signal(signal, raw[signal["code"]], today, cfg)
        except ValueError as exc:
            result = dict(public(signal), review_session=today, status="pending_data", source="minute_proxy", success=None, error=str(exc))
        reviewed.append(result)
    unique = {(r["signal_date"], r["code"], r.get("strategy_id")): r for r in outcomes}
    unique.update({(r["signal_date"], r["code"], r.get("strategy_id")): r for r in reviewed})
    outcomes = list(unique.values())
    save_json(history_path, outcomes)
    save_csv(root / "review" / (today + ".csv"), reviewed)
    save_csv(root / "learning_outcomes.csv", outcomes)
    rules = learn(outcomes, cfg, today)
    save_json(root / "learned_rules.json", rules)
    return rules, errors

def report(root, day, picks, status, cfg):
    folder = root / "daily" / day
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "status.json", status)
    save_csv(folder / "picks.csv", picks)
    lines = ["# 隔夜研究报告 " + day, "", "状态：" + status["state"],
             "生成时间：" + now().isoformat(), "",
             "运行说明：使用 efinance 接口，随时可运行；盘中获取实时，盘后获取收盘。",
             "", "## 候选（分数不代表胜率）", ""]
    for i, row in enumerate(picks, 1):
        lines += [f"{i}. {row['name']} {row['code']}｜{row['score']:.2f}/10｜参考价 {row['reference_price']}",
                  f"   行情时间 {row['signal_time']}；数据获取时间 {row['received_at']}",
                  "   理由：" + "；".join(row["reasons"]), "   复盘特征：" + "；".join(row["tags"]), ""]
    if not picks: lines += ["无可用候选。看状态和错误明细。", ""]
    lines += ["## 数据与学习", "", json.dumps(status, ensure_ascii=False, indent=2), ""]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("报告：", folder / "report.md", flush=True)

def scan_market(cfg=None, root=None):
    cfg = dict(DEFAULTS, **(cfg or {}))
    root = Path(root or Path(__file__).resolve().parent / "reports")
    day = now().date().isoformat()
    status = {"state": "初始化", "strategy_id": config_id(cfg), "errors": [], "version": VERSION}
    picks = []
    
    today_dt = datetime.fromisoformat(day)
    previous = (today_dt - timedelta(days=1)).date().isoformat()
    
    root.mkdir(parents=True, exist_ok=True)
    lock = root / (day + ".lock")
    try:
        lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("今日已有进程运行")
    os.close(lock_fd)

    try:
        report(root, day, [], status, cfg)
        save_json(root / "signals" / (day + ".json"), [])

        setup, errors = bounded_jobs([{"id": "spot", "kind": "spot"}], now() + timedelta(seconds=60), cfg)
        status["errors"] += errors

        rules, errors = review_previous(root, day, previous, cfg, now() + timedelta(seconds=180))
        status["errors"] += errors

        spot = setup.get("spot", {}).get("rows", [])
        if not spot:
            status["state"] = "实时快照不可用，停止选股"
            status["errors"].append({"error": "efinance 返回空"})
            return
        save_json(root / "raw" / (day + "_initial_spot.json"), setup["spot"])

        valid = [r for r in spot if number(r.get("涨跌幅")) is not None and number(r.get("成交额"), 0) > 0]
        breadth = sum(number(r["涨跌幅"]) > 0 for r in valid) / max(1, len(valid))
        status["market_breadth"] = breadth
        if breadth < cfg["min_market_breadth"]:
            status["state"] = "上涨占比低于阈值"
            return

        rows = shortlist(valid, cfg)
        status["shortlisted"] = len(rows)

        jobs = [{"id": r["代码"], "kind": "analyse", "code": r["代码"], "day": day,
                 "snapshot": r, "cfg": cfg, "breadth": breadth} for r in rows]
        initial, errors = bounded_jobs(jobs, now() + timedelta(seconds=300), cfg)
        status["errors"] += errors
        candidates = apply_learning([r for r in initial.values() if r], rules, cfg)
        
        if not candidates:
            status["state"] = "初筛无合格候选"
            return

        fresh_setup, errors = bounded_jobs([{"id": "spot", "kind": "spot"}], now() + timedelta(seconds=60), cfg)
        status["errors"] += errors
        fresh_rows = fresh_setup.get("spot", {}).get("rows", [])
        if not fresh_rows:
            status["state"] = "刷新失败"
            return
        save_json(root / "raw" / (day + "_final_spot.json"), fresh_setup["spot"])

        lookup = {code_of(r.get("代码")): r for r in shortlist(fresh_rows, dict(cfg, shortlist_size=10000))}
        jobs = [{"id": r["code"], "kind": "refresh", "code": r["code"], "day": day,
                 "snapshot": lookup[r["code"]], "history": r["_history"], "cfg": cfg, "breadth": breadth}
                for r in candidates[:cfg["refresh_size"]] if r["code"] in lookup]

        final, errors = bounded_jobs(jobs, now() + timedelta(seconds=180), dict(cfg, request_timeout_seconds=20))
        status["errors"] += errors
        candidates = apply_learning([r for r in final.values() if r], rules, cfg)

        picks = [public(r) for r in candidates if r["score"] >= cfg["min_score"]][:cfg["top_n"]]
        status["state"] = "候选已生成" if picks else "复核无可用候选"
        save_json(root / "signals" / (day + ".json"), picks)

    except Exception as exc:
        import traceback
        status["state"] = "运行失败"
        status["errors"].append({"error": type(exc).__name__ + ": " + str(exc), "traceback": traceback.format_exc()})
    finally:
        status["finished_at"] = now().isoformat()
        report(root, day, picks, status, cfg)
        lock.unlink(missing_ok=True)
    return picks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "reports"))
    args = parser.parse_args()
    cfg = dict(DEFAULTS, **load_json(args.config, {}))
    try:
        scan_market(cfg, args.output)
    except ValueError as exc:
        print(exc)

if __name__ == "__main__":
    mp.freeze_support()
    main()
