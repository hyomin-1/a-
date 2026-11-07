# -*- coding: utf-8 -*-
"""
A股猎手｜4万双模型回测：9:30开盘买入 vs 14:50尾盘买入

资金结构：
- 总资金 40,000
- 最多 2 只，每只 20,000
- 每只内部拆两批：
    10,000 在 +1% 止盈
    10,000 在 +2% 止盈
- 止损参考支撑位；止盈参考压力位做上限约束
- 选股看 MACD / RSI / MA10 / MA20 / 量比代理

运行方式：
    python backtest_dual.py --start 2025-11-07 --end 2026-11-07

程序会依次跑两个模型：
    1. open  模型：9:30 开盘价买入
    2. tail  模型：14:50 尾盘买入（用收盘价近似 + 额外滑点）
最后输出两个模型各自的汇总，方便直接对比。
"""

import argparse
import copy
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

import data_source

REPORT_ROOT = "reports/backtest_dual"
CACHE_DIR = "cache/history_dual"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def fnum(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def rsi_series(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.mask((down == 0) & (up > 0), 100)
    return out


def macd_series(series):
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


def build_features(df):
    if df is None or len(df) < 80:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in ("open", "close", "high", "low", "volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "close", "high", "low"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(df) < 80:
        return None

    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()
    df["ma10_up"] = df["ma10"] > df["ma10"].shift(1)
    df["ma20_slope3"] = df["ma20"] - df["ma20"].shift(3)

    df["dif"], df["dea"], df["mhist"] = macd_series(close)
    df["mhist_improve"] = df["mhist"] > df["mhist"].shift(1)
    df["rsi"] = rsi_series(close, 14)
    df["pct"] = close.pct_change() * 100

    df["low10"] = low.rolling(10).min()
    df["low20"] = low.rolling(20).min()
    df["low60"] = low.rolling(60).min()
    df["high10"] = high.rolling(10).max()
    df["high20"] = high.rolling(20).max()
    df["high60"] = high.rolling(60).max()

    df["vol5"] = vol.rolling(5).mean()
    df["volume_ratio"] = vol / df["vol5"].replace(0, np.nan)

    cand_sup = df[["low10", "low20", "low60", "ma20", "ma60"]]
    df["support"] = cand_sup.where(cand_sup.le(df["close"], axis=0)).max(axis=1)

    cand_res = df[["high10", "high20", "high60"]]
    df["resistance"] = cand_res.where(cand_res.ge(df["close"], axis=0)).min(axis=1)

    return df


def score_record(rec, config):
    score = 0.0
    reasons, risks = [], []

    price, pct = fnum(rec.get("close")), fnum(rec.get("pct"))
    ma10, ma20 = fnum(rec.get("ma10")), fnum(rec.get("ma20"))
    ma20_slope3 = fnum(rec.get("ma20_slope3"))
    dif, dea = fnum(rec.get("dif")), fnum(rec.get("dea"))
    rsi = fnum(rec.get("rsi"))
    support, resistance = fnum(rec.get("support")), fnum(rec.get("resistance"))
    volume_ratio = fnum(rec.get("volume_ratio"))

    if not np.isfinite(price) or not np.isfinite(pct):
        return None

    if np.isfinite(ma10) and price > ma10 and bool(rec.get("ma10_up")):
        score += 1.0
        reasons.append("价格站上MA10且MA10向上")
    elif np.isfinite(ma20) and price > ma20:
        score += 0.5
        reasons.append("价格仍在MA20上方")
    else:
        risks.append("价格位于MA20下方")

    if np.isfinite(ma20_slope3) and ma20_slope3 >= 0:
        score += 0.5
        reasons.append("MA20趋势未明显走弱")

    if np.isfinite(dif) and np.isfinite(dea) and dif > dea:
        score += 0.7
        reasons.append("日线MACD多头")
    elif bool(rec.get("mhist_improve")):
        score += 0.4
        reasons.append("日线MACD柱体改善")
    else:
        risks.append("日线MACD动能偏弱")

    if np.isfinite(rsi):
        if 45 <= rsi <= 68:
            score += 0.6
            reasons.append("日线RSI适中")
        elif rsi > 78:
            risks.append("日线RSI过热")

    if 0.2 <= pct <= 3.5:
        score += 0.7
        reasons.append("当日涨幅适中")
    elif -1.0 <= pct < 0.2 and np.isfinite(ma20) and price >= ma20:
        score += 0.4
        reasons.append("强势趋势内小幅整理")
    elif pct > 5:
        risks.append("当日涨幅偏大")

    support_gap = (price / support - 1) * 100 if np.isfinite(support) and support > 0 else np.nan
    resistance_gap = (resistance / price - 1) * 100 if np.isfinite(resistance) and price > 0 else np.nan

    if np.isfinite(support_gap) and 0 <= support_gap <= 3:
        score += 0.8
        reasons.append("价格靠近近端支撑")
    elif np.isfinite(support_gap) and support_gap > 7:
        risks.append("距离下方支撑较远")

    if np.isfinite(resistance_gap) and resistance_gap <= 1.2:
        score -= 0.6
        risks.append("上方压力位较近")
    elif np.isfinite(resistance_gap) and resistance_gap >= 2.5:
        score += 0.5
        reasons.append("上方有一定空间")

    if np.isfinite(volume_ratio):
        if 1.05 <= volume_ratio <= 2.5:
            score += 0.5
            reasons.append("成交量温和活跃")
        elif volume_ratio > 4:
            risks.append("成交量异常放大")

    score = max(0.0, min(10.0, score))
    return score, "；".join(reasons), "；".join(risks) if risks else "未发现明显技术风险"


def screen_record(rec, config, args):
    code, name = str(rec["code"]), str(rec["name"])
    price, pct = fnum(rec.get("close")), fnum(rec.get("pct"))

    if not np.isfinite(price) or price <= 0 or not np.isfinite(pct):
        return None

    scr = config.get("screening", {})
    if price < float(scr.get("min_price", 0)) or price > float(scr.get("max_price", 800)):
        return None
    if pct < float(scr.get("min_day_pct", -10)) or pct <= -10:
        return None

    lim = limit_rate(code, name)
    if pct <= -lim + 0.2:
        return None
    if scr.get("exclude_limit_up", True) and pct >= lim - 0.2:
        return None

    item = score_record(rec, config)
    if item is None:
        return None

    score, reasons, risks = item
    if score < float(args.min_score):
        return None
    return score, reasons, risks


def load_history_cached(code, config, args, hist_config):
    os.makedirs(args.cache_dir, exist_ok=True)
    path = os.path.join(args.cache_dir, f"{code}.csv")
    start_ts = pd.Timestamp(args.start).normalize()
    end_ts = pd.Timestamp(args.end).normalize()

    if os.path.exists(path) and not args.refresh:
        try:
            old = pd.read_csv(path)
            old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.normalize()
            old = old.dropna(subset=["date"])
            if len(old) >= 80 and old["date"].max() >= end_ts - pd.Timedelta(days=10) and \
               old["date"].min() <= start_ts - pd.Timedelta(days=100):
                return old
        except Exception:
            pass

    df = data_source.get_history(code, hist_config)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def load_histories(universe, config, args, hist_config):
    histories, errors = {}, []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(load_history_cached, s["code"], config, args, hist_config): s
            for s in universe
        }
        for i, future in enumerate(as_completed(futures), 1):
            s = futures[future]
            try:
                df = future.result()
                if df is not None and len(df) >= 80:
                    histories[s["code"]] = (s["name"], df)
            except Exception as exc:
                errors.append({"代码": s["code"], "名称": s["name"], "错误": str(exc)[:300]})
            if i % 50 == 0 or i == len(futures):
                log(f"历史数据进度 {i}/{len(futures)}，成功 {len(histories)}")

    return histories, errors


def build_day_map(histories, args):
    start_ts = pd.Timestamp(args.start).normalize()
    end_ts = pd.Timestamp(args.end).normalize()
    day_map = defaultdict(dict)

    for code, (name, df) in histories.items():
        feat = build_features(df)
        if feat is None:
            continue

        for r in feat.to_dict("records"):
            d = pd.Timestamp(r["date"]).normalize()
            if d < start_ts or d > end_ts:
                continue

            day_map[d][code] = {
                "code": code, "name": name, "date": d,
                "open": fnum(r.get("open")),
                "high": fnum(r.get("high")),
                "low": fnum(r.get("low")),
                "close": fnum(r.get("close")),
                "pct": fnum(r.get("pct")),
                "ma10": fnum(r.get("ma10")),
                "ma20": fnum(r.get("ma20")),
                "ma10_up": bool(r.get("ma10_up")) if pd.notna(r.get("ma10_up")) else False,
                "ma20_slope3": fnum(r.get("ma20_slope3")),
                "dif": fnum(r.get("dif")),
                "dea": fnum(r.get("dea")),
                "mhist_improve": bool(r.get("mhist_improve")) if pd.notna(r.get("mhist_improve")) else False,
                "rsi": fnum(r.get("rsi")),
                "support": fnum(r.get("support")),
                "resistance": fnum(r.get("resistance")),
                "volume_ratio": fnum(r.get("volume_ratio")),
            }
    return day_map


def run_one_model(config, args, mode):
    """mode: 'open' 或 'tail'。两个模型共用选股和资金规则，只改买入价。"""
    model_dir = os.path.join(REPORT_ROOT, mode)
    os.makedirs(model_dir, exist_ok=True)

    start_ts = pd.Timestamp(args.start).normalize()
    end_ts = pd.Timestamp(args.end).normalize()

    hist_config = copy.deepcopy(config)
    hist_config.setdefault("history", {})
    hist_config["history"]["history_days"] = max(int(args.history_days), 450)
    hist_config["history"]["minimum_history_rows"] = 80

    log(f"===== 开始模型：{mode} =====")
    data_source.reset_data_source_status()
    universe = data_source.get_stock_universe(config)
    if args.max_stocks and args.max_stocks > 0:
        universe = universe[:args.max_stocks]

    histories, errors = load_histories(universe, config, args, hist_config)
    if not histories:
        raise RuntimeError(f"{mode} 模型没有取得历史数据")

    day_map = build_day_map(histories, args)
    trade_dates = sorted(day_map.keys())
    if len(trade_dates) < 2:
        raise RuntimeError(f"{mode} 模型交易日不足")

    cash = float(args.cash)
    initial_cash = float(args.cash)
    positions = []
    trades = []
    equity_rows = []

    commission_rate = float(args.commission)
    stamp_rate = float(args.stamp)
    base_slippage = float(args.slippage)
    tail_extra_slippage = float(args.tail_extra_slippage) if mode == "tail" else 0.0
    max_positions = int(args.max_positions)
    per_stock_cash = initial_cash / max_positions  # 每只 2 万

    def buy_price_for(rec):
        if mode == "open":
            raw = fnum(rec.get("open"))
            if not np.isfinite(raw) or raw <= 0:
                raw = fnum(rec.get("close"))
            return raw * (1 + base_slippage) if np.isfinite(raw) and raw > 0 else np.nan
        else:
            raw = fnum(rec.get("close"))
            return raw * (1 + base_slippage + tail_extra_slippage) if np.isfinite(raw) and raw > 0 else np.nan

    def do_sell(pos, date, raw_price, reason, di, batch_name, batch_shares):
        nonlocal cash
        if not np.isfinite(raw_price) or raw_price <= 0 or batch_shares <= 0:
            return

        sell_price = raw_price * (1 - base_slippage)
        amount = sell_price * batch_shares
        fee = max(5.0, amount * commission_rate) + amount * stamp_rate
        proceeds = amount - fee
        cash += proceeds

        pnl = proceeds - pos["batch_cost"][batch_name]
        pnl_pct = pnl / pos["batch_cost"][batch_name] * 100 if pos["batch_cost"][batch_name] > 0 else np.nan

        trades.append({
            "模型": mode,
            "代码": pos["code"],
            "名称": pos["name"],
            "批次": batch_name,
            "买入日期": pos["buy_date"],
            "买入价": round(pos["buy_price"], 4),
            "卖出日期": date.strftime("%Y-%m-%d"),
            "卖出价": round(sell_price, 4),
            "股数": batch_shares,
            "买入成本": round(pos["batch_cost"][batch_name], 2),
            "卖出净额": round(proceeds, 2),
            "盈亏": round(pnl, 2),
            "盈亏%": round(pnl_pct, 2) if np.isfinite(pnl_pct) else None,
            "持有交易日": di - pos["buy_idx"],
            "卖出原因": reason,
            "买入评分": pos["score"],
            "支撑位": round(pos["support"], 4) if np.isfinite(pos["support"]) else None,
            "压力位": round(pos["resistance"], 4) if np.isfinite(pos["resistance"]) else None,
        })

    for di, date in enumerate(trade_dates):
        today = day_map.get(date, {})
        sold_today_codes = set()

        for pos in positions:
            rec = today.get(pos["code"])
            if rec:
                pos["last_close"] = fnum(rec.get("close"), pos["last_close"])

        for pos in list(positions):
            rec = today.get(pos["code"])
            if not rec:
                continue

            days_held = di - pos["buy_idx"]
            if days_held < 1:
                continue

            o = fnum(rec.get("open"))
            h = fnum(rec.get("high"))
            l = fnum(rec.get("low"))
            c = fnum(rec.get("close"))
            if not np.isfinite(c):
                continue

            # 批次 1：1万，+1% 止盈
            if not pos["batch_done"]["b1"]:
                tp1 = pos["batch_targets"]["b1"]
                stop_price = pos["stop"]
                raw_exit, reason = None, None

                if np.isfinite(o) and o <= stop_price:
                    raw_exit, reason = o, "止损跳空"
                elif np.isfinite(o) and o >= tp1:
                    raw_exit, reason = o, "止盈+1%跳空"
                elif np.isfinite(l) and l <= stop_price:
                    raw_exit, reason = stop_price, "止损"
                elif np.isfinite(h) and h >= tp1:
                    raw_exit, reason = tp1, "止盈+1%"

                if raw_exit is not None:
                    do_sell(pos, date, raw_exit, reason, di, "b1", pos["shares"]["b1"])
                    pos["batch_done"]["b1"] = True

            # 批次 2：1万，+2% 止盈
            if not pos["batch_done"]["b2"] and not pos["batch_done"]["b1"]:
                # 批次 2 只在批次 1 未触发时继续；如果批次 1 已卖，批次 2 仍独立判断
                pass

            if not pos["batch_done"]["b2"]:
                tp2 = pos["batch_targets"]["b2"]
                stop_price = pos["stop"]
                raw_exit, reason = None, None

                if np.isfinite(o) and o <= stop_price:
                    raw_exit, reason = o, "止损跳空"
                elif np.isfinite(o) and o >= tp2:
                    raw_exit, reason = o, "止盈+2%跳空"
                elif np.isfinite(l) and l <= stop_price:
                    raw_exit, reason = stop_price, "止损"
                elif np.isfinite(h) and h >= tp2:
                    raw_exit, reason = tp2, "止盈+2%"

                if raw_exit is not None:
                    do_sell(pos, date, raw_exit, reason, di, "b2", pos["shares"]["b2"])
                    pos["batch_done"]["b2"] = True

            # 两批都完成，移除持仓
            if pos["batch_done"]["b1"] and pos["batch_done"]["b2"]:
                positions.remove(pos)
                sold_today_codes.add(pos["code"])

        # 买入：收盘/开盘选股，空几个仓位买几个
        slots = max_positions - len(positions)
        if slots > 0 and today:
            held = {p["code"] for p in positions}
            cands = []

            for code, rec in today.items():
                if code in held or code in sold_today_codes:
                    continue
                item = screen_record(rec, config, args)
                if item is not None:
                    cands.append((item[0], fnum(rec.get("pct"), 0), code, rec))

            cands.sort(key=lambda x: (x[0], x[1]), reverse=True)

            for score, pct, code, rec in cands:
                if len(positions) >= max_positions:
                    break

                bp = buy_price_for(rec)
                if not np.isfinite(bp) or bp <= 0:
                    continue

                # 每只股票预算 2 万；如果现金不足，按可用现金买
                budget = min(cash, per_stock_cash)
                if budget < bp * 100:
                    continue

                support = fnum(rec.get("support"))
                resistance = fnum(rec.get("resistance"))

                # 两批各自目标价
                b1_target = bp * 1.01
                b2_target = bp * 1.02

                # 压力位约束：如果压力位更近，把止盈压到压力位下方
                if np.isfinite(resistance) and resistance > bp * 1.001:
                    b1_target = min(b1_target, resistance * 0.998)
                    b2_target = min(b2_target, resistance * 0.998)

                # 止损参考支撑位
                stop = bp * (1 - float(args.stop_loss_pct) / 100.0)
                if np.isfinite(support) and 0 < support < bp * 0.999:
                    stop = max(stop, support * 0.997)
                if stop >= bp:
                    stop = bp * (1 - float(args.stop_loss_pct) / 100.0)

                total_shares = int(budget / bp / 100) * 100
                if total_shares < 200:
                    continue

                b1_shares = (total_shares // 200) * 100
                b2_shares = total_shares - b1_shares
                if b1_shares < 100 or b2_shares < 100:
                    continue

                # 扣现金
                total_cost = 0.0
                for sh in (b1_shares, b2_shares):
                    amt = bp * sh
                    fee = max(5.0, amt * commission_rate)
                    total_cost += amt + fee

                if total_cost > cash:
                    # 减少一手再试
                    if total_shares >= 300:
                        total_shares -= 100
                        b1_shares = (total_shares // 200) * 100
                        b2_shares = total_shares - b1_shares
                        total_cost = 0.0
                        for sh in (b1_shares, b2_shares):
                            amt = bp * sh
                            fee = max(5.0, amt * commission_rate)
                            total_cost += amt + fee
                    if total_cost > cash or b1_shares < 100 or b2_shares < 100:
                        continue

                cash -= total_cost
                positions.append({
                    "code": code, "name": rec["name"],
                    "buy_date": date.strftime("%Y-%m-%d"), "buy_idx": di,
                    "buy_price": bp,
                    "shares": {"b1": b1_shares, "b2": b2_shares},
                    "batch_cost": {
                        "b1": bp * b1_shares + max(5.0, bp * b1_shares * commission_rate),
                        "b2": bp * b2_shares + max(5.0, bp * b2_shares * commission_rate),
                    },
                    "batch_targets": {"b1": b1_target, "b2": b2_target},
                    "batch_done": {"b1": False, "b2": False},
                    "stop": stop,
                    "last_close": fnum(rec.get("close")),
                    "support": support,
                    "resistance": resistance,
                    "score": round(score, 2),
                })

        mv = 0.0
        for p in positions:
            for b in ("b1", "b2"):
                if not p["batch_done"][b]:
                    mv += p["shares"][b] * p["last_close"]
        equity_rows.append({
            "日期": date.strftime("%Y-%m-%d"),
            "现金": round(cash, 2),
            "持仓市值": round(mv, 2),
            "总权益": round(cash + mv, 2),
            "持仓数": len(positions),
        })

    # 结束强制平仓
    last_date, last_di = trade_dates[-1], len(trade_dates) - 1
    for pos in list(positions):
        for b in ("b1", "b2"):
            if not pos["batch_done"][b]:
                do_sell(pos, last_date, pos["last_close"], "回测结束平仓", last_di, b, pos["shares"][b])
                pos["batch_done"][b] = True
        positions.remove(pos)

    if equity_rows:
        equity_rows[-1]["现金"] = round(cash, 2)
        equity_rows[-1]["持仓市值"] = 0.0
        equity_rows[-1]["总权益"] = round(cash, 2)
        equity_rows[-1]["持仓数"] = 0

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    final_equity = float(equity_df["总权益"].iloc[-1]) if not equity_df.empty else initial_cash
    total_return = (final_equity / initial_cash - 1) * 100 if initial_cash > 0 else np.nan

    calendar_days = max(1, (end_ts - start_ts).days)
    annualized = ((final_equity / initial_cash) ** (365.0 / calendar_days) - 1) * 100 \
        if initial_cash > 0 and final_equity > 0 else np.nan

    max_dd = 0.0
    peak = -np.inf
    if not equity_df.empty:
        for eq in equity_df["总权益"]:
            peak = max(peak, eq)
            if peak > 0:
                max_dd = min(max_dd, eq / peak - 1)
    max_dd *= 100

    summary = {
        "模型": mode,
        "回测区间": f"{args.start} ~ {args.end}",
        "初始资金": round(initial_cash, 2),
        "最终权益": round(final_equity, 2),
        "总收益率%": round(total_return, 2) if np.isfinite(total_return) else None,
        "年化收益率%": round(annualized, 2) if np.isfinite(annualized) else None,
        "最大回撤%": round(max_dd, 2),
        "交易批次数": len(trades_df),
        "买入滑点": f"{base_slippage:.4%}" + (f" + 尾盘额外 {tail_extra_slippage:.4%}" if mode == "tail" else ""),
        "佣金率": f"{commission_rate:.4%}",
        "卖出印花税": f"{stamp_rate:.4%}",
    }

    if len(trades_df) > 0:
        pnl_pct = pd.to_numeric(trades_df["盈亏%"], errors="coerce")
        valid = pnl_pct.dropna()
        summary["胜率%"] = round((valid > 0).sum() / len(valid) * 100, 2) if len(valid) else None
        summary["平均每批盈亏%"] = round(valid.mean(), 2) if len(valid) else None
        summary["盈利批次平均%"] = round(valid[valid > 0].mean(), 2) if (valid > 0).any() else None
        summary["亏损批次平均%"] = round(valid[valid < 0].mean(), 2) if (valid < 0).any() else None

    trades_df.to_csv(os.path.join(model_dir, "trades.csv"), index=False, encoding="utf-8-sig")
    equity_df.to_csv(os.path.join(model_dir, "equity.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(model_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if errors:
        pd.DataFrame(errors).to_csv(
            os.path.join(model_dir, "data_errors.csv"), index=False, encoding="utf-8-sig"
        )

    log(f"{mode} 模型完成：" + json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--cash", type=float, default=40000, help="总资金，默认40000")
    parser.add_argument("--max-positions", type=int, default=2)
    parser.add_argument("--stop-loss-pct", type=float, default=1.5, help="止损百分比，默认1.5")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--max-stocks", type=int, default=0, help="限制股票池数量，0=全市场")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--history-days", type=int, default=450)
    parser.add_argument("--slippage", type=float, default=0.0005, help="基础单边滑点")
    parser.add_argument("--tail-extra-slippage", type=float, default=0.001,
                        help="尾盘模型额外滑点，默认0.10%")
    parser.add_argument("--commission", type=float, default=0.00025)
    parser.add_argument("--stamp", type=float, default=0.0005)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--refresh", action="store_true")

    args = parser.parse_args()

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")
    if args.start is None:
        args.start = (pd.Timestamp(args.end) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    if args.workers is None:
        args.workers = int(config.get("data_source", {}).get("max_workers", 8))
    if args.min_score is None:
        args.min_score = float(config.get("screening", {}).get("min_score", 0))

    os.makedirs(REPORT_ROOT, exist_ok=True)
    summaries = []

    for mode in ("open", "tail"):
        try:
            s = run_one_model(config, args, mode)
            summaries.append(s)
        except Exception as exc:
            log(f"{mode} 模型失败：{type(exc).__name__}: {exc}")
            traceback.print_exc()

    if summaries:
        cmp_df = pd.DataFrame(summaries)
        cmp_df.to_csv(os.path.join(REPORT_ROOT, "compare.csv"), index=False, encoding="utf-8-sig")
        log("\n========== 双模型对比 ==========")
        log(cmp_df.to_string(index=False))
        log(f"\n对比结果已保存：{os.path.join(REPORT_ROOT, 'compare.csv')}")


if __name__ == "__main__":
    main()
