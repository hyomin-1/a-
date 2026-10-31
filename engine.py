# -*- coding: utf-8 -*-
"""A股猎手高级版（包含资金流、技术指标评分与复盘）"""
import os
import re
import json
import time
import traceback
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TZ = ZoneInfo("Asia/Shanghai")
REPORT_DIR = "reports"
MAX_WORKERS = int(os.getenv("A_SHARE_WORKERS", "4"))
TOP_N = 3
MIN_SCORE = 5.5
HISTORY_DAYS = 45
TOP_INDUSTRY_BOARDS = 12
TOP_CONCEPT_BOARDS = 12

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
    if len(df) < 20:
        return None
    close = df[close_col]
    df["_close"] = close
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA10_PREV"] = df["MA10"].shift(1)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    df["RSI"] = calculate_rsi(close, 14)
    high_col = get_col(df, ["high", "最高"])
    low_col = get_col(df, ["low", "最低"])
    open_col = get_col(df, ["open", "开盘"])
    if high_col and low_col:
        df["_high"] = pd.to_numeric(df[high_col], errors="coerce")
        df["_low"] = pd.to_numeric(df[low_col], errors="coerce")
    if open_col:
        df["_open"] = pd.to_numeric(df[open_col], errors="coerce")
    return df

def _try_sector_flow(indicator="今日", sector_type="行业资金流"):
    fn = getattr(ak, "stock_sector_fund_flow_rank", None)
    if not callable(fn):
        return None
    try:
        df = fn(indicator=indicator, sector_type=sector_type)
    except TypeError:
        df = fn(indicator=indicator)
    return df

def _find_flow_col(df):
    if df is None: return None
    patterns = [r"今日主力净流入.*净额", r"主力净流入.*净额", r"主力净流入", r"净流入.*净额", r"净流入"]
    for pattern in patterns:
        for col in df.columns:
            if re.search(pattern, str(col)):
                return col
    return None

def _find_board_name_col(df):
    if df is None: return None
    return get_col(df, ["名称", "板块名称", "行业名称", "概念名称", "板块"])

def fetch_board_flow():
    status = {"行业": "未获取", "概念": "未获取"}
    board_rows = []
    code_map = {}
    for board_type, status_key, constituent_func_name, limit in [
        ("行业", "行业", "stock_board_industry_cons_em", TOP_INDUSTRY_BOARDS),
        ("概念", "概念", "stock_board_concept_cons_em", TOP_CONCEPT_BOARDS),
    ]:
        try:
            sector_type = "行业资金流" if board_type == "行业" else "概念资金流"
            raw = _try_sector_flow("今日", sector_type=sector_type)
            if raw is None or raw.empty:
                continue
            name_col = _find_board_name_col(raw)
            flow_col = _find_flow_col(raw)
            if not name_col or not flow_col:
                continue
            raw = raw.copy()
            raw["_flow"] = pd.to_numeric(
                raw[flow_col].astype(str).str.replace(",", "", regex=False)
                .str.replace("亿", "", regex=False).str.replace("万", "", regex=False),
                errors="coerce"
            )
            raw = raw.dropna(subset=["_flow"]).sort_values("_flow", ascending=False)
            top = raw.head(limit)
            status[status_key] = f"成功（读取{len(raw)}个板块）"
            cons_fn = getattr(ak, constituent_func_name, None)
            for rank, (_, item) in enumerate(top.iterrows(), start=1):
                board_name = str(item[name_col])
                flow_value = safe_float(item["_flow"])
                pct_col = get_col(raw, ["今日涨跌幅", "涨跌幅", "涨跌幅(%)", "板块涨跌幅"])
                board_pct = safe_float(item[pct_col]) if pct_col else np.nan
                board_rows.append({
                    "板块类型": board_type, "板块名称": board_name,
                    "资金流排名": rank, "资金流值": flow_value, "板块涨幅%": board_pct,
                })
                if not callable(cons_fn): continue
                try:
                    members = cons_fn(symbol=board_name)
                    if members is None or members.empty: continue
                    code_col = get_col(members, ["代码", "股票代码", "证券代码"])
                    if not code_col: continue
                    for code_raw in members[code_col].astype(str):
                        code = normalize_code(code_raw)
                        if len(code) != 6: continue
                        board_score = (limit - rank + 1) / limit
                        if board_type == "行业": board_score *= 1.0
                        else: board_score *= 0.75
                        old = code_map.get(code, {"score": 0.0, "boards": []})
                        old["score"] += board_score
                        old["boards"].append(f"{board_type}:{board_name}(第{rank})")
                        code_map[code] = old
                except Exception: pass
        except Exception: pass
    if board_rows:
        boards = pd.DataFrame(board_rows).drop_duplicates(subset=["板块类型", "板块名称"], keep="first")
    else:
        boards = pd.DataFrame(columns=["板块类型", "板块名称", "资金流排名", "资金流值", "板块涨幅%"])
    return boards, code_map, status

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
            if not df.empty and "代码" in df.columns:
                df["代码"] = df["代码"].map(normalize_code)
            return stamp, df
        except Exception: pass
    daily = []
    for filename in os.listdir(REPORT_DIR):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.csv", filename):
            date_text = filename[:-4]
            if date_text < today:
                daily.append((date_text, os.path.join(REPORT_DIR, filename)))
    if not daily: return None, pd.DataFrame()
    prev_date, path = sorted(daily, reverse=True)[0]
    try:
        df = pd.read_csv(path, dtype={"代码": str})
        if not df.empty and "代码" in df.columns:
            df["代码"] = df["代码"].map(normalize_code)
        return prev_date, df
    except Exception: return prev_date, pd.DataFrame()

def learning_paths():
    return {
        "outcomes": os.path.join(REPORT_DIR, "learning_outcomes.csv"),
        "memory": os.path.join(REPORT_DIR, "learning_memory.json"),
        "rules": os.path.join(REPORT_DIR, "learned_rules.json"),
    }

def read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
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
    change = (current_price / old_price - 1) * 100 if old_price > 0 and current_price > 0 else np.nan
    if np.isfinite(change) and change <= -2:
        if np.isfinite(market_pct) and market_pct <= -1: reasons.append("市场整体走弱")
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
            "代码": code, "名称": str(old.get("名称", "")),
            "入选评分": safe_float(old.get("评分")), "入选价": old_price,
            "复盘价": current_price, "T1收益%": round(change, 3),
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
            rules["规则"][f"{col}={bucket}"] = {
                "样本数": int(len(group)), "明显下跌率": round(loss_rate, 3),
                "平均T1收益%": round(avg_return, 3), "建议扣分": round(penalty, 2)
            }
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
    notes = [f"已更新学习样本：本轮新增{len(outcome_rows)}条，累计{len(new_df)}条。"]
    return new_df, notes

def apply_learned_penalties(candidates):
    paths = learning_paths()
    rules = read_json(paths["rules"], {}).get("规则", {})
    if not rules or not candidates: return candidates
    for item in candidates:
        features = {
            "RSI区间": "偏热" if safe_float(item.get("RSI")) >= 70 else "非偏热",
            "乖离区间": "高乖离" if safe_float(item.get("偏离MA10%")) >= 5 else "普通乖离",
            "量比区间": "高量比" if safe_float(item.get("量比")) >= 3 else "普通量比",
            "换手区间": "高换手" if safe_float(item.get("换手率")) >= 15 else "普通换手",
            "MACD区间": "零轴下" if safe_float(item.get("MACD")) < 0 else "零轴上",
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
    start_date = (sh_now() - timedelta(days=HISTORY_DAYS)).strftime("%Y%m%d")
    last_error = None
    for attempt in range(3):
        try:
            hist = ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start_date, end_date=today_compact, adjust="qfq"
            )
            if hist is not None and len(hist) >= 20: return hist
            last_error = RuntimeError("历史行情为空或不足20条")
        except Exception as exc:
            last_error = exc
        if attempt < 2: time.sleep(1)
    raise RuntimeError(f"历史行情获取失败：{last_error}")

def process_single_stock(row, today_compact, board_map=None):
    raw_code = str(row.get("代码", ""))
    code = normalize_code(raw_code)
    name = str(row.get("名称", ""))
    if not code: return None
    try:
        hist_raw = fetch_hist(code, today_compact)
        hist = calculate_indicators(hist_raw)
        if hist is None or len(hist) < 20: raise RuntimeError("有效历史K线不足20条")
        last = hist.iloc[-1]
        prev = hist.iloc[-2]
        price = safe_float(row.get("最新价"))
        pct = safe_float(row.get("涨跌幅"))
        if not np.isfinite(price) or price <= 0: raise RuntimeError("实时价格无效")
        if not np.isfinite(pct): pct = safe_float(last.get("涨跌幅"), 0.0)
        ma10 = safe_float(last.get("MA10"))
        ma5 = safe_float(last.get("MA5"))
        macd = safe_float(last.get("MACD"))
        prev_macd = safe_float(prev.get("MACD"))
        rsi = safe_float(last.get("RSI"))
        dif = safe_float(last.get("DIF"))
        dea = safe_float(last.get("DEA"))
        if not np.isfinite(ma10) or not np.isfinite(macd): raise RuntimeError("技术指标缺失")
        volume_ratio = safe_float(row.get("量比"), 0.0)
        weibi = safe_float(row.get("委比"), np.nan)
        outer = safe_float(row.get("外盘"), np.nan)
        inner = safe_float(row.get("内盘"), np.nan)
        turnover = safe_float(row.get("换手率"), np.nan)
        amount = safe_float(row.get("成交额"), np.nan)
        score = 0.0
        reasons = []
        risks = []
        if price > ma10: score += 1.2; reasons.append("价格在MA10上方")
        else: risks.append("价格低于MA10")
        if np.isfinite(ma5) and price >= ma5: score += 0.4
        if np.isfinite(ma10) and np.isfinite(safe_float(last.get("MA10_PREV"))) and ma10 >= safe_float(last.get("MA10_PREV")): score += 0.4
        else: risks.append("MA10未上行")
        if macd > prev_macd: score += 0.7; reasons.append("MACD较前一日改善")
        else: risks.append("MACD未改善")
        if macd > 0: score += 0.4
        if dif > dea and safe_float(prev.get("DIF")) <= safe_float(prev.get("DEA")): score += 0.4; reasons.append("近端MACD金叉")
        elif dif > dea: score += 0.2
        if 50 <= rsi <= 68: score += 1.0; reasons.append("RSI处于偏强但未明显过热区")
        elif 45 <= rsi < 50 or 68 < rsi <= 73: score += 0.5
        elif rsi > 78: risks.append("RSI过热")
        if 1.15 <= volume_ratio <= 2.8: score += 0.7; reasons.append("量比适中")
        elif 0.9 <= volume_ratio < 1.15: score += 0.35
        elif volume_ratio > 4: risks.append("量比过高")
        if 0.5 <= pct <= 3.5: score += 0.5
        elif -1.5 <= pct < 0.5 and price >= ma10: score += 0.4; reasons.append("强势趋势内回调/横盘")
        elif pct > 6: risks.append("当日涨幅较大")
        if np.isfinite(turnover):
            if 2 <= turnover <= 12: score += 0.3
            elif turnover > 20: risks.append("换手率过高")
        if np.isfinite(weibi):
            if weibi > 10: score += 0.3
            elif weibi < -20: risks.append("委比偏弱")
        if np.isfinite(outer) and np.isfinite(inner) and (outer + inner) > 0:
            if outer > inner: score += 0.5
            else: risks.append("外盘弱于内盘")
        board_info = (board_map or {}).get(code, {})
        board_score_raw = safe_float(board_info.get("score"), 0.0)
        if board_score_raw > 0:
            sector_points = min(1.7, board_score_raw / 1.75 * 1.7)
            score += sector_points
            reasons.append("属于当日资金流靠前板块")
            boards_text = "、".join(board_info.get("boards", [])[:3])
        else:
            boards_text = "未映射到当日资金流靠前板块"
        recent = hist.tail(10)
        high_col = get_col(recent, ["_high", "high", "最高"])
        low_col = get_col(recent, ["_low", "low", "最低"])
        high10 = safe_float(recent[high_col].max()) if high_col else np.nan
        low10 = safe_float(recent[low_col].min()) if low_col else np.nan
        if np.isfinite(high10) and high10 > 0 and np.isfinite(low10) and high10 > low10:
            pos10 = (price - low10) / (high10 - low10)
            if 0.25 <= pos10 <= 0.72: score += 0.9; reasons.append("处于近10日相对中低位")
            elif pos10 < 0.25 and price >= ma10: score += 0.6; reasons.append("低位尝试企稳")
            elif pos10 > 0.88: risks.append("接近近10日高位")
        deviation = (price / ma10 - 1) * 100 if ma10 > 0 else np.nan
        if np.isfinite(deviation):
            if -1 <= deviation <= 4: score += 0.6
            elif deviation > 7: risks.append("明显偏离MA10")
        penalty = 0.0
        if "当日涨幅较大" in risks: penalty += 0.8
        if "RSI过热" in risks: penalty += 0.5
        if "量比过高" in risks: penalty += 0.4
        if "换手率过高" in risks: penalty += 0.4
        if "接近近10日高位" in risks: penalty += 0.5
        if "明显偏离MA10" in risks: penalty += 0.5
        score = max(0.0, min(10.0, score - penalty))
        if "ST" in name.upper() or "退" in name: return None
        if score < MIN_SCORE: return None
        target_price = round(price * 1.03, 2)
        stop_loss = round(price * 0.98, 2)
        date_col = get_col(hist, ["date", "日期"])
        hist_date = str(last[date_col])[:10] if date_col else "未知"
        return {
            "代码": code, "名称": name, "最新价": round(price, 3), "涨幅%": round(pct, 2),
            "成交额": round(amount, 0) if np.isfinite(amount) else np.nan,
            "换手率": round(turnover, 2) if np.isfinite(turnover) else np.nan,
            "量比": round(volume_ratio, 2),
            "委比": round(weibi, 1) if np.isfinite(weibi) else np.nan,
            "外/内": round(outer / inner, 2) if np.isfinite(outer) and np.isfinite(inner) and inner > 0 else np.nan,
            "RSI": round(rsi, 1) if np.isfinite(rsi) else np.nan, "MACD": round(macd, 4), "MA10": round(ma10, 3),
            "偏离MA10%": round(deviation, 2) if np.isfinite(deviation) else np.nan,
            "板块": boards_text, "评分": round(score, 2),
            "评分理由": "；".join(reasons[:5]) or "综合指标评分",
            "风险提示": "；".join(risks) if risks else "未触发主要风险项",
            "历史K线日期": hist_date, "次日参考止盈": target_price, "参考止损": stop_loss,
        }
    except Exception as exc:
        return {"_error": f"{code} {name}: {type(exc).__name__}: {exc}", "_code": code}

def get_spot_snapshot():
    try:
        spot = ak.stock_zh_a_spot_em()
        source = "stock_zh_a_spot_em"
    except Exception as first_exc:
        log(f"东财快照失败尝试备用：{first_exc}")
        spot = ak.stock_zh_a_spot()
        source = "stock_zh_a_spot"
    if spot is None or spot.empty: raise RuntimeError("实时快照为空")
    spot = spot.copy()
    for col in ["最新价", "涨跌幅", "成交额", "量比", "委比", "外盘", "内盘", "换手率"]:
        if col in spot.columns: spot[col] = pd.to_numeric(spot[col], errors="coerce")
        else: spot[col] = np.nan
    return spot, source

def review_previous_picks(previous_date, previous_df, current_spot):
    if previous_df is None or previous_df.empty: return pd.DataFrame(), ["没有找到更早日期的推荐"]
    current = current_spot.copy()
    current["代码"] = current["代码"].map(normalize_code)
    price_col = "最新价"
    current = current.drop_duplicates("代码").set_index("代码")
    records = []
    notes = [f"上次推荐日期：{previous_date}"]
    for _, old in previous_df.iterrows():
        code = normalize_code(old.get("代码", ""))
        name = str(old.get("名称", ""))
        old_price = safe_float(old.get("最新价"))
        old_score = safe_float(old.get("评分"))
        if code in current.index:
            new_price = safe_float(current.loc[code].get(price_col))
            pct_now = safe_float(current.loc[code].get("涨跌幅"))
            change = (new_price / old_price - 1) * 100 if old_price > 0 and new_price > 0 else np.nan
            if np.isfinite(change):
                if change >= 2: judgment = "价格表现偏强"
                elif change > -1: judgment = "价格暂时横盘/小幅波动"
                else: judgment = "价格走弱，重新评估"
            else: judgment = "价格数据不完整"
            _, diagnosis = diagnose_drop(old, new_price, pct_now)
            records.append({
                "上次推荐日期": previous_date, "代码": code, "名称": name,
                "上次评分": old_score, "上次参考价": old_price, "当前价": new_price,
                "相对上次价格变化%": round(change, 2) if np.isfinite(change) else np.nan,
                "当前日涨幅%": pct_now, "复盘判断": judgment, "下跌归因候选": diagnosis,
            })
        else:
            records.append({
                "上次推荐日期": previous_date, "代码": code, "名称": name,
                "上次评分": old_score, "上次参考价": old_price,
                "当前价": np.nan, "相对上次价格变化%": np.nan,
                "当前日涨幅%": np.nan, "复盘判断": "当前快照未找到该股票",
            })
    return pd.DataFrame(records), notes

def write_report(today, candidates_count, result, previous_date, review_df, board_df, statuses, scan_stats):
    ensure_dirs()
    lines = [
        f"# A股猎手升级版报告：{today}", "",
        f"- 运行时间：{sh_now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 扫描快照接口：{scan_stats.get('snapshot_source', '未知')}",
        f"- 基础过滤后股票数：{scan_stats.get('universe', 0)}",
        f"- 历史行情成功计算数：{scan_stats.get('success', 0)}",
        f"- 单股失败数：{scan_stats.get('failed', 0)}",
        f"- 达到最低评分 {MIN_SCORE} 分的候选数：{candidates_count}",
        f"- 本次推荐数：{len(result)}（最多 {TOP_N} 只）", "",
        "## 一、上次推荐复盘",
    ]
    if previous_date and review_df is not None and not review_df.empty:
        lines.append(f"上次推荐日期：{previous_date}")
        for _, row in review_df.iterrows():
            delta = row.get("相对上次价格变化%")
            delta_text = f"{delta:+.2f}%" if pd.notna(delta) else "无法计算"
            lines.append(f"- {row.get('名称')}（{row.get('代码')}）：上次评分 {row.get('上次评分')}，参考价 {row.get('上次参考价')}，当前价 {row.get('当前价')}，相对变化 {delta_text}；判断：{row.get('复盘判断')}；归因候选：{row.get('下跌归因候选', '暂无')}")
    else:
        lines.append("- 暂无历史推荐记录。")

    lines.extend(["", "## 二、当日板块资金流"])
    for key, value in statuses.items(): lines.append(f"- {key}板块数据：{value}")
    if board_df is not None and not board_df.empty:
        for _, row in board_df.sort_values(["板块类型", "资金流排名"]).iterrows():
            flow = row.get("资金流值")
            flow_text = f"{flow:,.2f}" if pd.notna(flow) else "未知"
            pct = row.get("板块涨幅%")
            pct_text = f"{pct:.2f}%" if pd.notna(pct) else "未知"
            lines.append(f"- {row.get('板块类型')}第{int(row.get('资金流排名'))}：{row.get('板块名称')}；接口资金流值 {flow_text}；板块涨幅 {pct_text}")
    else:
        lines.append("- 未获得可用板块资金流排名；本次个股评分中的板块因子可能缺失。")

    lines.extend(["", "## 三、本次候选股票"])
    if result is None or result.empty:
        lines.append(f"- 没有股票达到 {MIN_SCORE} 分；这不代表全市场没有机会，避免为了凑数强行推荐。")
    else:
        for i, item in enumerate(result.to_dict("records"), start=1):
            lines.extend([
                f"### {i}. {item['名称']}（{item['代码']}）｜评分 {item['评分']}/10",
                f"- 当前价：{item['最新价']}；当日涨幅：{item['涨幅%']}%",
                f"- 板块：{item.get('板块', '未知')}",
                f"- 入选理由：{item.get('评分理由', '无')}",
                f"- 风险提示：{item.get('风险提示', '无')}",
                f"- 情景参考价（+3%）：{item.get('次日参考止盈', '未知')}；参考止损价（-2%）：{item.get('参考止损', '未知')}",
                f"- 指标：MA10={item.get('MA10')}，MACD={item.get('MACD')}，RSI={item.get('RSI')}，量比={item.get('量比')}",
                "",
            ])

    lines.extend([
        "", "## 四、持续学习系统",
        "- 每次运行会把前一日推荐与当前价格匹配，保存T+1结果、下跌归因候选和特征分组。",
        "- reports/learning_outcomes.csv：历史结果样本；reports/learning_memory.json：学习摘要；reports/learned_rules.json：达到样本门槛的自适应规则。",
        "- 规则至少需要8个同类样本才可能扣分，且最多扣1.5分；不会因为单只股票一次下跌就直接封杀。",
        "- 当前自动归因主要基于入选时保存的技术与量价信号。公告/政策/新闻的真实因果需要额外接入可靠新闻或公告源。",
        "", "## 五、数据质量与风险说明",
        "- 个股失败明细写入 reports/scan_errors.csv；接口失败不会被解释为“无股票符合条件”。",
        "- 当日资金流是数据源的估算口径，不等同于真实主力身份或未来收益保证。",
        "- 次日参考止盈/止损为情景参考价；A股跳空、涨跌停、滑点和交易费用可能导致无法按该价格成交。",
        "- 建议累计至少15–20个交易日的推荐记录后，再评估胜率、平均收益、最大回撤并考虑调整权重。",
    ])
    with open(os.path.join(REPORT_DIR, "report.md"), "w", encoding="utf-8") as f: f.write("\n".join(lines))

def scan_market():
    started = time.time()
    ensure_dirs()
    now = sh_now()
    today = now.strftime("%Y-%m-%d")
    today_compact = now.strftime("%Y%m%d")
    log("获取全市场实时快照...")
    spot, snapshot_source = get_spot_snapshot()
    spot_all = spot.copy()

    required = ["代码", "名称", "最新价", "涨跌幅", "成交额"]
    missing = [col for col in required if col not in spot.columns]
    if missing: raise RuntimeError(f"实时快照缺少必要字段：{missing}")

    spot = spot.dropna(subset=["最新价", "涨跌幅", "成交额"]).copy()
    spot["纯代码"] = spot["代码"].map(normalize_code)

    name_series = spot["名称"].astype(str)
    code_series = spot["纯代码"].astype(str)
    
    # 过滤：剔除ST、退市、北交所、科创板，且价格、涨幅、成交额达标
    condition_pct = np.where(code_series.str.startswith("3"), spot["涨跌幅"] < 19.5, spot["涨跌幅"] < 9.5)
    spot = spot[
        (~name_series.str.contains(r"ST|退", case=False, na=False))
        & (~code_series.str.startswith(("8", "4", "9", "688")))
        & (spot["最新价"] > 2)
        & (spot["最新价"] <= 1000)
        & (spot["成交额"] >= 20_000_000)
        & condition_pct
    ].copy()

    log(f"基础过滤完成：{len(spot)} 只股票。获取当日行业/概念资金流排名...")
    board_df, board_map, board_status = fetch_board_flow()
    
    previous_date, previous_df = load_previous_picks(today)
    review_df, review_notes = review_previous_picks(previous_date, previous_df, spot_all)
    learning_df, learning_notes = update_learning(previous_date, previous_df, spot_all, today)
    
    candidates = []
    errors = []
    total = len(spot)
    success = 0
    failed = 0
    log(f"开始深度计算；并发数={MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_stock, row, today_compact, board_map): row for _, row in spot.iterrows()}
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
            if completed % 200 == 0 or completed == total:
                log(f"进度：{completed}/{total}；候选 {len(candidates)}；失败 {failed}")

    error_path = os.path.join(REPORT_DIR, "scan_errors.csv")
    if errors:
        pd.DataFrame(errors).to_csv(error_path, mode="a", header=not os.path.exists(error_path), index=False, encoding="utf-8-sig")
    
    if board_df is not None and not board_df.empty:
        board_df.to_csv(os.path.join(REPORT_DIR, f"sector_flow_{today}.csv"), index=False, encoding="utf-8-sig")
    if review_df is not None and not review_df.empty:
        review_df.to_csv(os.path.join(REPORT_DIR, f"review_{today}.csv"), index=False, encoding="utf-8-sig")

    if candidates:
        candidates = apply_learned_penalties(candidates)
        df_all = pd.DataFrame(candidates).sort_values(["评分", "量比"], ascending=[False, False])
        result = df_all.head(TOP_N).copy()
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
        result = pd.DataFrame()
        pd.DataFrame(columns=["代码", "名称", "评分"]).to_csv(os.path.join(REPORT_DIR, f"{today}.csv"), index=False, encoding="utf-8-sig")

    if result is not None and not result.empty:
        run_dir = os.path.join(REPORT_DIR, "runs")
        os.makedirs(run_dir, exist_ok=True)
        run_stamp = sh_now().strftime("%Y-%m-%d_%H%M%S")
        result.to_csv(os.path.join(run_dir, f"{run_stamp}.csv"), index=False, encoding="utf-8-sig")

    stats = {"snapshot_source": snapshot_source, "universe": total, "success": success, "failed": failed, "seconds": round(time.time() - started, 1)}
    write_report(today, len(candidates), result, previous_date, review_df, board_df, board_status, stats)
    log(f"运行完成：候选 {len(candidates)}，最终推荐 {len(result)}，失败 {failed}，耗时 {stats['seconds']} 秒。")

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
