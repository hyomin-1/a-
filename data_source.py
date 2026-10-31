# -*- coding: utf-8 -*-
import json
import re
import time
import logging
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

logger = logging.getLogger("ashare_data_source")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Connection": "keep-alive"
})

SOURCE_STATUS = {"errors": []}

def _record_error(task: str, error: Any) -> None:
    SOURCE_STATUS["errors"].append({"task": str(task), "error": str(error)[:500]})

def _request(url: str, params: Optional[dict] = None, timeout: int = 15, retry_times: int = 2, encoding: Optional[str] = None) -> str:
    last_error = None
    for attempt in range(max(1, retry_times + 1)):
        try:
            response = SESSION.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            if encoding: response.encoding = encoding
            else: response.encoding = response.apparent_encoding or "utf-8"
            return response.text
        except Exception as exc:
            last_error = exc
            if attempt < retry_times: time.sleep(1)
    raise RuntimeError(f"请求失败：{last_error}")

def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or str(value).strip() in ("-", "--", "None", "null", ""): return default
        result = float(value)
        return result if result == result else default
    except (TypeError, ValueError): return default

def _fetch_tencent_quotes(codes: List[str]) -> Dict[str, dict]:
    if not codes: return {}
    symbols = [("sh" if c.startswith("6") else "sz") + c for c in codes]
    url = "https://qt.gtimg.cn/q=" + ",".join(symbols)
    content = _request(url, encoding="gb18030")

    result = {}
    for line in content.splitlines():
        match = re.search(r'v_(sh\d{6}|sz\d{6})="(.*?)";?', line)
        if not match: continue
        symbol, values_str = match.group(1), match.group(2)
        values = values_str.split("~")
        if len(values) < 35: continue

        code = symbol[2:]
        price = _to_float(values[3], 0.0)
        prev_close = _to_float(values[4], 0.0)

        if price <= 0 or prev_close <= 0: continue

        result[code] = {
            "code": code, "name": values[1].strip(), "price": price,
            "prev_close": prev_close, "open": _to_float(values[5], price) or price,
            "high": _to_float(values[33], price) or price, "low": _to_float(values[34], price) or price,
            "amount": _to_float(values[37], 0.0) or 0.0,
            "change_pct": round((price / prev_close - 1) * 100, 4)
        }
    return result

def get_market_snapshot(config: dict) -> List[dict]:
    # 生成主板和创业板的代码池
    codes = []
    for prefix in ["600", "601", "603", "605", "000", "001", "002", "003", "300", "301"]:
        for i in range(1000):
            codes.append(f"{prefix}{i:03d}")
    
    batch_size = 80
    workers = 4
    batches = [codes[i:i + batch_size] for i in range(0, len(codes), batch_size)]
    
    snapshot = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_tencent_quotes, batch): batch for batch in batches}
        for future in as_completed(futures):
            try: snapshot.update(future.result())
            except Exception as exc: _record_error("tencent_batch", exc)

    if len(snapshot) < 1000:
        raise RuntimeError(f"行情数量异常：{len(snapshot)}")

    result = list(snapshot.values())
    result.sort(key=lambda x: _to_float(x.get("amount"), 0.0) or 0.0, reverse=True)
    return result

def get_history(code: str, config: dict) -> pd.DataFrame:
    symbol = ("sh" if code.startswith("6") else "sz") + code
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{symbol},day,,,220,qfq"}
    
    content = _request(url, params=params)
    payload = json.loads(content)
    data = payload.get("data", {}).get(symbol, {})
    rows = data.get("qfqday") or data.get("day") or []

    if not rows: raise RuntimeError(f"没有返回 {code} 的历史K线")

    records = []
    for row in rows:
        if len(row) < 6: continue
        records.append({
            "date": row[0], "open": _to_float(row[1]), "close": _to_float(row[2]),
            "high": _to_float(row[3]), "low": _to_float(row[4]), "volume": _to_float(row[5]) or 0,
        })

    frame = pd.DataFrame(records)
    if frame.empty: raise RuntimeError(f"{code} 历史K线解析为空")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return frame
