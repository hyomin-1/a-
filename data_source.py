# -*- coding: utf-8 -*-
"""A股猎手行情数据模块：腾讯主源、新浪备用、腾讯历史K线。"""
import json
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

log = logging.getLogger("ashare_data")
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
})

STATUS = {
    "universe_source": "sina",
    "realtime_primary": "tencent",
    "realtime_fallback": "sina",
    "history_source": "tencent",
    "universe_count": 0,
    "snapshot_count": 0,
    "tencent_count": 0,
    "sina_count": 0,
    "errors": []
}


def request_text(url, params=None, timeout=15, retries=1, encoding=None):
    last = None
    for attempt in range(retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            r.encoding = encoding or r.apparent_encoding or "utf-8"
            return r.text
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(1 + attempt)
    raise RuntimeError(str(last))


def code6(value):
    digits = re.sub(r"\D", "", str(value))
    if len(digits) < 6:
        raise ValueError("无效股票代码: " + str(value))
    return digits[-6:]


def market_symbol(code):
    code = code6(code)
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh" + code
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return "sz" + code
    raise ValueError("不支持的沪深A股代码: " + code)


def allowed(code, cfg):
    code = code6(code)
    m = cfg.get("market", {})
    if code.startswith(("600", "601", "603", "605")):
        return m.get("include_shanghai", True)
    if code.startswith(("688", "689")):
        return m.get("include_shanghai", True) and m.get("include_star_market", True)
    if code.startswith(("000", "001", "002", "003")):
        return m.get("include_shenzhen", True)
    if code.startswith(("300", "301")):
        return m.get("include_shenzhen", True) and m.get("include_chinext", True)
    return False


def valid_name(name, cfg):
    name = str(name or "").strip()
    if not name:
        return False
    m = cfg.get("market", {})
    if m.get("exclude_st", True) and "ST" in name.upper():
        return False
    if m.get("exclude_delisting", True) and ("退" in name or "摘牌" in name):
        return False
    return True


def num(value, default=None):
    try:
        if value is None or str(value).strip() in ("", "-", "--", "None"):
            return default
        x = float(value)
        return x if pd.notna(x) else default
    except Exception:
        return default


def get_stock_universe(config):
    """新浪分页股票列表只用于取得代码和名称，不作为实时行情主源。"""
    sc = config.get("data_source", {})
    page_size = int(sc.get("sina_batch_size", 80))
    url = (
        "https://vip.stock.finance.sina.com.cn/"
        "quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
    )
    stocks = {}
    empty = 0

    for page in range(1, 101):
        params = {
            "page": page, "num": page_size, "sort": "symbol",
            "asc": 1, "node": "hs_a", "symbol": "", "_s_r_a": "page"
        }
        raw = request_text(
            url, params=params,
            timeout=int(sc.get("request_timeout_seconds", 15)),
            retries=int(sc.get("retry_times", 1))
        )
        try:
            rows = json.loads(raw)
        except Exception:
            match = re.search(r"(\[.*\])", raw, re.S)
            if not match:
                raise RuntimeError(f"新浪股票列表第{page}页不是有效JSON")
            rows = json.loads(match.group(1))

        if not rows:
            empty += 1
            if empty >= 2:
                break
            continue

        empty = 0
        for row in rows:
            code = re.sub(r"\D", "", str(row.get("code", "")))
            name = str(row.get("name", "")).strip()
            if len(code) != 6 or not allowed(code, config) or not valid_name(name, config):
                continue
            stocks[code] = {"code": code, "name": name}

    if len(stocks) < 1000:
        raise RuntimeError(f"股票列表数量异常：仅取得{len(stocks)}只，停止扫描")
    STATUS["universe_count"] = len(stocks)
    return list(stocks.values())


def parse_tencent_line(line):
    m = re.search(r'v_(sh\d{6}|sz\d{6})="(.*?)";?', line)
    if not m:
        return None
    symbol, raw = m.group(1), m.group(2)
    a = raw.split("~")
    if len(a) < 35:
        return None

    code = symbol[2:]
    name = a[1].strip()
    price = num(a[3])
    prev = num(a[4])
    if not price or not prev or price <= 0 or prev <= 0:
        return None

    # 腾讯字段：3现价、4昨收、5开盘、6成交量、7外盘、8内盘、
    # 33最高、34最低。成交额字段版本可能变化，缺失时不猜测。
    amount = num(a[37]) if len(a) > 37 else None
    volume = num(a[6], 0)
    return {
        "code": code, "name": name, "price": price, "prev_close": prev,
        "open": num(a[5], price), "high": num(a[33], price),
        "low": num(a[34], price), "volume": volume,
        "amount": amount, "outer": num(a[7]), "inner": num(a[8]),
        "change_pct": (price / prev - 1) * 100,
        "turnover_rate": None, "source": "tencent"
    }


def fetch_tencent_quotes(codes, config):
    if not codes:
        return {}
    sc = config.get("data_source", {})
    size = int(sc.get("tencent_batch_size", 80))
    result = {}
    for i in range(0, len(codes), size):
        batch = codes[i:i + size]
        symbols = ",".join(market_symbol(c) for c in batch)
        url = "https://qt.gtimg.cn/q=" + symbols
        try:
            raw = request_text(
                url, timeout=int(sc.get("request_timeout_seconds", 15)),
                retries=int(sc.get("retry_times", 1)), encoding="gb18030"
            )
            for line in raw.splitlines():
                item = parse_tencent_line(line)
                if item:
                    result[item["code"]] = item
        except Exception as exc:
            STATUS["errors"].append("腾讯行情批次失败: " + str(exc)[:200])
    return result


def fetch_sina_quotes(codes, config):
    """只补腾讯未返回的代码；新浪403时停止备用请求。"""
    if not codes:
        return {}
    sc = config.get("data_source", {})
    size = int(sc.get("sina_batch_size", 80))
    result = {}
    for i in range(0, len(codes), size):
        batch = codes[i:i + size]
        symbols = ",".join(market_symbol(c) for c in batch)
        try:
            raw = request_text(
                "https://hq.sinajs.cn/list=" + symbols,
                timeout=int(sc.get("request_timeout_seconds", 15)),
                retries=0, encoding="gb18030"
            )
            for line in raw.splitlines():
                m = re.search(r'hq_str_(sh\d{6}|sz\d{6})="(.*?)";?', line)
                if not m:
                    continue
                code = m.group(1)[2:]
                a = m.group(2).split(",")
                if len(a) < 10:
                    continue
                price, prev = num(a[3]), num(a[2])
                if not price or not prev or price <= 0 or prev <= 0:
                    continue
                result[code] = {
                    "code": code, "name": a[0], "price": price,
                    "prev_close": prev, "open": num(a[1], price),
                    "high": num(a[4], price), "low": num(a[5], price),
                    "volume": num(a[8], 0), "amount": num(a[9]),
                    "outer": None, "inner": None,
                    "change_pct": (price / prev - 1) * 100,
                    "turnover_rate": None, "source": "sina"
                }
        except Exception as exc:
            STATUS["errors"].append("新浪备用行情失败: " + str(exc)[:200])
            if "403" in str(exc) or "Forbidden" in str(exc):
                break
    return result


def get_market_snapshot(config):
    universe = get_stock_universe(config)
    codes = [x["code"] for x in universe]
    quotes = fetch_tencent_quotes(codes, config)
    STATUS["tencent_count"] = len(quotes)

    missing = [c for c in codes if c not in quotes]
    fallback = fetch_sina_quotes(missing, config)
    STATUS["sina_count"] = len(fallback)
    for code, item in fallback.items():
        quotes.setdefault(code, item)

    name_map = {x["code"]: x["name"] for x in universe}
    result = []
    for code, item in quotes.items():
        item["name"] = item.get("name") or name_map.get(code, "")
        if valid_name(item["name"], config):
            result.append(item)

    STATUS["snapshot_count"] = len(result)
    min_ratio = float(config.get("data_source", {}).get("minimum_snapshot_ratio", 0.65))
    coverage = len(result) / max(1, len(codes))
    if coverage < min_ratio:
        raise RuntimeError(
            f"实时行情覆盖率只有{coverage:.1%}，低于要求{min_ratio:.1%}，停止本轮扫描"
        )
    return result


def get_history(code, config):
    """腾讯前复权日K线。"""
    code = code6(code)
    symbol = market_symbol(code)
    hc = config.get("history", {})
    sc = config.get("data_source", {})
    count = int(hc.get("history_days", 180))
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    raw = request_text(
        url, params={"param": f"{symbol},day,,,{count},qfq"},
        timeout=int(sc.get("request_timeout_seconds", 15)),
        retries=int(sc.get("retry_times", 1))
    )
    payload = json.loads(raw)
    node = payload.get("data", {}).get(symbol, {})
    rows = node.get("qfqday") or node.get("day") or []
    records = []
    for row in rows:
        if len(row) < 6:
            continue
        records.append({
            "date": row[0], "open": num(row[1]), "close": num(row[2]),
            "high": num(row[3]), "low": num(row[4]),
            "volume": num(row[5], 0),
            "amount": num(row[6]) if len(row) > 6 else None
        })
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"{code}历史日K线为空")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open", "close", "high", "low", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close", "high", "low"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(df) < int(hc.get("minimum_history_rows", 60)):
        raise RuntimeError(f"{code}历史K线不足60条")
    return df


def get_data_source_status():
    return dict(STATUS)


def reset_data_source_status():
    STATUS.update({
        "universe_count": 0, "snapshot_count": 0,
        "tencent_count": 0, "sina_count": 0, "errors": []
    })
