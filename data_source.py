
# -*- coding: utf-8 -*-
"""
A股猎手 - 行情数据模块

实时行情：
1. 腾讯财经为主（适配GitHub Actions环境）
2. 新浪财经为备用（遇403停止重试）

历史K线：
腾讯财经

不使用东方财富，不使用 AkShare。

对外提供：
- get_stock_universe(config)
- get_market_snapshot(config)
- get_history(code, config)
- get_data_source_status()
"""

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
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Connection": "keep-alive"
})

SOURCE_STATUS = {
    "universe_source": None,
    "realtime_primary": "tencent",
    "realtime_fallback": "sina",
    "history_source": "tencent",
    "universe_count": 0,
    "snapshot_count": 0,
    "sina_count": 0,
    "tencent_count": 0,
    "errors": []
}


def _record_error(task: str, error: Any) -> None:
    """记录数据接口错误，避免错误被静默忽略。"""
    SOURCE_STATUS["errors"].append({
        "task": str(task),
        "error": str(error)[:500]
    })

    logger.warning("%s: %s", task, error)


def _request(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 15,
    retry_times: int = 2,
    encoding: Optional[str] = None
) -> str:
    """统一网络请求，支持有限次数重试。"""

    last_error = None

    for attempt in range(max(1, retry_times + 1)):
        try:
            response = SESSION.get(
                url,
                params=params,
                timeout=timeout
            )

            response.raise_for_status()

            if encoding:
                response.encoding = encoding
            elif "sinajs.cn" in url:
                response.encoding = "gb18030"
            else:
                response.encoding = response.apparent_encoding or "utf-8"

            return response.text

        except Exception as exc:
            last_error = exc

            if attempt < retry_times:
                time.sleep(1.0 + attempt)

    raise RuntimeError(f"请求失败：{last_error}")


def _market_config(config: dict) -> dict:
    return config.get("market", {})


def _source_config(config: dict) -> dict:
    return config.get("data_source", {})


def _history_config(config: dict) -> dict:
    return config.get("history", {})


def _normalize_code(code: str) -> str:
    """将股票代码统一成六位数字。"""

    value = str(code).strip().lower()

    value = value.replace("sh", "").replace("sz", "")
    value = value.replace("bj", "")

    digits = re.sub(r"\D", "", value)

    if len(digits) != 6:
        raise ValueError(f"无效股票代码：{code}")

    return digits


def _sina_symbol(code: str) -> str:
    """生成新浪行情代码。"""

    code = _normalize_code(code)

    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh" + code

    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return "sz" + code

    raise ValueError(f"暂不支持的股票市场代码：{code}")


def _tencent_symbol(code: str) -> str:
    """生成腾讯行情代码。"""

    code = _normalize_code(code)

    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh" + code

    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return "sz" + code

    raise ValueError(f"暂不支持的股票市场代码：{code}")


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        text = str(value).strip()

        if not text or text in ("-", "--", "None", "null"):
            return None

        result = float(text)

        if result != result:
            return None

        return result

    except (TypeError, ValueError):
        return None


def _is_valid_stock_name(name: str, market_config: dict) -> bool:
    name = str(name or "").strip()

    if not name:
        return False

    if market_config.get("exclude_st", True):
        if "ST" in name.upper():
            return False

    if market_config.get("exclude_delisting", True):
        invalid_words = ("退市", "退")
        if any(word in name for word in invalid_words):
            return False

    return True


def _is_allowed_code(code: str, market_config: dict) -> bool:
    """判断股票代码是否属于允许扫描的市场。"""

    code = str(code).strip()

    if not re.fullmatch(r"\d{6}", code):
        return False

    if code.startswith(("600", "601", "603", "605")):
        return bool(market_config.get("include_shanghai", True))

    if code.startswith(("688", "689")):
        return (
            bool(market_config.get("include_shanghai", True))
            and bool(market_config.get("include_star_market", True))
        )

    if code.startswith(("000", "001", "002", "003")):
        return bool(market_config.get("include_shenzhen", True))

    if code.startswith(("300", "301")):
        return (
            bool(market_config.get("include_shenzhen", True))
            and bool(market_config.get("include_chinext", True))
        )

    # 本版本默认不扫描北交所。
    if code.startswith(("4", "8", "9")):
        return bool(market_config.get("include_beijing", False))

    return False


def _parse_sina_market_row(row: dict) -> Optional[dict]:
    """
    解析新浪市场列表中的一只股票。

    新浪市场列表常见字段：
    symbol、code、name、trade、settlement、
    open、high、low、volume、amount、turnoverratio。
    """

    code = str(row.get("code") or "").strip()

    if not code:
        symbol = str(row.get("symbol") or "").strip()
        code = re.sub(r"\D", "", symbol)

    if not re.fullmatch(r"\d{6}", code):
        return None

    name = str(row.get("name") or "").strip()

    price = _to_float(row.get("trade"))
    prev_close = _to_float(row.get("settlement"))
    open_price = _to_float(row.get("open"))
    high = _to_float(row.get("high"))
    low = _to_float(row.get("low"))
    volume = _to_float(row.get("volume"))
    amount = _to_float(row.get("amount"))

    if not price or price <= 0:
        return None

    if not prev_close or prev_close <= 0:
        return None

    change_pct = (price / prev_close - 1) * 100

    return {
        "code": code,
        "name": name,
        "price": price,
        "prev_close": prev_close,
        "open": open_price or price,
        "high": high or price,
        "low": low or price,
        "volume": volume or 0,
        "amount": amount or 0,
        "change_pct": round(change_pct, 4),
        "turnover_rate": _to_float(row.get("turnoverratio")),
        "source": "sina"
    }


def _fetch_sina_universe_page(
    page: int,
    page_size: int,
    config: dict
) -> List[dict]:

    source_config = _source_config(config)

    url = (
        "https://vip.stock.finance.sina.com.cn/"
        "quotes_service/api/json_v2.php/"
        "Market_Center.getHQNodeData"
    )

    params = {
        "page": page,
        "num": page_size,
        "sort": "symbol",
        "asc": 1,
        "node": "hs_a",
        "symbol": "",
        "_s_r_a": "page"
    }

    content = _request(
        url,
        params=params,
        timeout=int(source_config.get("request_timeout_seconds", 15)),
        retry_times=int(source_config.get("retry_times", 2))
    )

    try:
        data = json.loads(content)
    except Exception:
        # 个别响应可能带有额外空白或非标准包裹。
        match = re.search(r"(\[.*\])", content, re.S)

        if not match:
            raise RuntimeError(
                f"新浪市场列表第 {page} 页不是有效JSON"
            )

        data = json.loads(match.group(1))

    if not isinstance(data, list):
        raise RuntimeError(
            f"新浪市场列表第 {page} 页返回格式异常"
        )

    return data


def get_stock_universe(config: dict) -> List[dict]:
    """
    获取沪深A股股票列表。

    使用新浪市场列表接口分页获取。
    不使用本地写死的股票名单。

    如果列表获取不完整，不把部分股票冒充全市场。
    """

    market_config = _market_config(config)
    source_config = _source_config(config)

    page_size = int(source_config.get("sina_batch_size", 80))
    max_pages = 100
    empty_pages_to_stop = 2

    stocks = {}
    empty_pages = 0
    successful_pages = 0

    SOURCE_STATUS["universe_source"] = "sina"

    for page in range(1, max_pages + 1):

        try:
            rows = _fetch_sina_universe_page(
                page,
                page_size,
                config
            )

            successful_pages += 1

            if not rows:
                empty_pages += 1

                if empty_pages >= empty_pages_to_stop:
                    break

                continue

            empty_pages = 0

            for row in rows:
                if not isinstance(row, dict):
                    continue

                item = _parse_sina_market_row(row)

                if not item:
                    continue

                code = item["code"]

                if not _is_allowed_code(code, market_config):
                    continue

                if not _is_valid_stock_name(
                    item["name"],
                    market_config
                ):
                    continue

                stocks[code] = item

        except Exception as exc:
            _record_error(f"sina_universe_page_{page}", exc)

            # 中间页失败不能继续冒充完整市场。
            raise RuntimeError(
                f"新浪全市场列表第 {page} 页获取失败，"
                "本轮无法确认全市场覆盖率"
            ) from exc

    result = list(stocks.values())

    if successful_pages < 2 or len(result) < 1000:
        raise RuntimeError(
            "新浪股票列表数量异常："
            f"成功页数={successful_pages}，"
            f"有效股票数={len(result)}"
        )

    SOURCE_STATUS["universe_count"] = len(result)

    return result


def _fetch_sina_quotes(
    codes: List[str],
    config: dict
) -> Dict[str, dict]:
    """批量获取新浪实时行情。"""

    if not codes:
        return {}

    source_config = _source_config(config)

    symbols = [_sina_symbol(code) for code in codes]

    url = "https://hq.sinajs.cn/list=" + ",".join(symbols)

    content = _request(
        url,
        timeout=int(source_config.get("request_timeout_seconds", 15)),
        retry_times=int(source_config.get("retry_times", 2)),
        encoding="gb18030"
    )

    result = {}

    for line in content.splitlines():

        match = re.search(
            r'hq_str_(sh\d{6}|sz\d{6})="(.*?)";?',
            line
        )

        if not match:
            continue

        symbol = match.group(1)
        values = match.group(2).split(",")

        if len(values) < 10:
            continue

        code = symbol[2:]

        name = values[0].strip()
        open_price = _to_float(values[1])
        prev_close = _to_float(values[2])
        price = _to_float(values[3])
        high = _to_float(values[4])
        low = _to_float(values[5])
        volume = _to_float(values[8])
        amount = _to_float(values[9])

        if not price or price <= 0:
            continue

        if not prev_close or prev_close <= 0:
            continue

        result[code] = {
            "code": code,
            "name": name,
            "price": price,
            "prev_close": prev_close,
            "open": open_price or price,
            "high": high or price,
            "low": low or price,
            "volume": volume or 0,
            "amount": amount or 0,
            "change_pct": round(
                (price / prev_close - 1) * 100,
                4
            ),
            "turnover_rate": None,
            "source": "sina"
        }

    return result


def _fetch_tencent_quotes(
    codes: List[str],
    config: dict
) -> Dict[str, dict]:
    """腾讯行情备用接口。"""

    if not codes:
        return {}

    source_config = _source_config(config)

    symbols = [_tencent_symbol(code) for code in codes]

    url = "https://qt.gtimg.cn/q=" + ",".join(symbols)

    content = _request(
        url,
        timeout=int(source_config.get("request_timeout_seconds", 15)),
        retry_times=int(source_config.get("retry_times", 2)),
        encoding="gb18030"
    )

    result = {}

    for line in content.splitlines():

        match = re.search(
            r'v_(sh\d{6}|sz\d{6})="(.*?)";?',
            line
        )

        if not match:
            continue

        symbol = match.group(1)
        values = match.group(2).split("~")

        # 腾讯行情常见字段：
        # 1名称、2代码、3现价、4昨收、5今开、
        # 6成交量、7外盘、8内盘、33最高、34最低、
        # 37成交额（字段随接口版本可能扩展）。
        if len(values) < 35:
            continue

        code = symbol[2:]

        name = values[1].strip()
        price = _to_float(values[3])
        prev_close = _to_float(values[4])
        open_price = _to_float(values[5])
        volume = _to_float(values[6])
        high = _to_float(values[33])
        low = _to_float(values[34])

        # 腾讯字段位置可能变化。
        # 成交额只有在字段存在且有效时才使用。
        amount = None

        if len(values) > 37:
            amount = _to_float(values[37])

        if not price or price <= 0:
            continue

        if not prev_close or prev_close <= 0:
            continue

        result[code] = {
            "code": code,
            "name": name,
            "price": price,
            "prev_close": prev_close,
            "open": open_price or price,
            "high": high or price,
            "low": low or price,
            "volume": volume or 0,
            "amount": amount or 0,
            "change_pct": round(
                (price / prev_close - 1) * 100,
                4
            ),
            "turnover_rate": None,
            "source": "tencent"
        }

    return result


def _chunks(items: List[str], size: int):
    size = max(1, int(size))

    for index in range(0, len(items), size):
        yield items[index:index + size]


def get_market_snapshot(config: dict) -> List[dict]:
    """获取全市场实时快照：腾讯优先，新浪备用。

    新浪 hq.sinajs.cn 在 GitHub Actions 环境出现 403 时，
    不再反复请求；直接使用腾讯行情补齐，避免整轮扫描被新浪阻断。
    """
    source_config = _source_config(config)
    universe = get_stock_universe(config)
    codes = []
    for item in universe:
        try:
            code = _normalize_code(item.get("code", ""))
            if _is_allowed_code(code, _market_config(config)):
                codes.append(code)
        except (TypeError, ValueError):
            continue
    codes = list(dict.fromkeys(codes))
    if not codes:
        raise RuntimeError("股票代码列表为空，停止扫描")

    batch_size = int(source_config.get("tencent_batch_size", 80))
    workers = max(1, int(source_config.get("max_workers", 4)))
    batches = list(_chunks(codes, batch_size))
    snapshot = {}

    def fetch_batch(batch):
        return _fetch_tencent_quotes(batch, config)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                data = future.result()
                snapshot.update(data)
                missing = [code for code in batch if code not in data]
                if missing:
                    _record_error("tencent_missing_quotes", f"缺少 {len(missing)} 只")
            except Exception as exc:
                _record_error("tencent_realtime_batch", exc)

    SOURCE_STATUS["realtime_primary"] = "tencent"
    SOURCE_STATUS["realtime_fallback"] = "sina"
    SOURCE_STATUS["tencent_count"] = len(snapshot)

    # 腾讯未返回的股票才尝试新浪；新浪403时记录一次失败，不影响已取得的腾讯行情。
    missing_codes = [code for code in codes if code not in snapshot]
    sina_size = int(source_config.get("sina_batch_size", 80))
    sina_result = {}
    for batch in _chunks(missing_codes, sina_size):
        try:
            sina_result.update(_fetch_sina_quotes(batch, config))
        except Exception as exc:
            message = str(exc)
            _record_error("sina_fallback_batch", message)
            if "403" in message or "Forbidden" in message:
                logger.warning("新浪接口返回403，停止后续新浪备用请求")
                break

    for code, item in sina_result.items():
        snapshot.setdefault(code, item)
    SOURCE_STATUS["sina_count"] = len(sina_result)
    SOURCE_STATUS["snapshot_count"] = len(snapshot)
    SOURCE_STATUS["universe_count"] = len(codes)

    coverage = len(snapshot) / len(codes)
    minimum_ratio = float(source_config.get("minimum_snapshot_ratio", 0.70))
    if coverage < minimum_ratio:
        raise RuntimeError(
            "全市场行情覆盖率不足，停止本轮选股。"
            f"股票总数={len(codes)}，有效行情={len(snapshot)}，"
            f"覆盖率={coverage:.1%}，最低要求={minimum_ratio:.1%}。"
            "请检查腾讯接口或股票列表接口是否可访问。"
        )

    universe_map = {item["code"]: item for item in universe if item.get("code")}
    result = []
    for code, item in snapshot.items():
        merged = dict(item)
        original = universe_map.get(code, {})
        if not merged.get("name"):
            merged["name"] = original.get("name", "")
        if merged.get("turnover_rate") is None:
            merged["turnover_rate"] = original.get("turnover_rate")
        result.append(merged)
    result.sort(key=lambda x: _to_float(x.get("amount"), 0) or 0, reverse=True)
    return result


def get_history(
    code: str,
    config: dict
) -> pd.DataFrame:
    """
    获取腾讯前复权日K线。

    返回字段：
    date, open, close, high, low, volume, amount
    """

    code = _normalize_code(code)
    symbol = _tencent_symbol(code)

    history_config = _history_config(config)
    source_config = _source_config(config)

    count = int(
        history_config.get("history_days", 220)
    )

    url = (
        "https://web.ifzq.gtimg.cn/"
        "appstock/app/fqkline/get"
    )

    params = {
        "param": f"{symbol},day,,, {count},qfq".replace(" ", "")
    }

    content = _request(
        url,
        params=params,
        timeout=int(source_config.get("request_timeout_seconds", 15)),
        retry_times=int(source_config.get("retry_times", 2))
    )

    try:
        payload = json.loads(content)
    except Exception as exc:
        raise RuntimeError(
            f"腾讯历史K线JSON解析失败：{code}"
        ) from exc

    data = payload.get("data", {}).get(symbol, {})

    rows = (
        data.get("qfqday")
        or data.get("day")
        or []
    )

    if not rows:
        raise RuntimeError(
            f"腾讯没有返回 {code} 的历史K线"
        )

    records = []

    for row in rows:

        if not isinstance(row, list) or len(row) < 6:
            continue

        records.append({
            "date": row[0],
            "open": _to_float(row[1]),
            "close": _to_float(row[2]),
            "high": _to_float(row[3]),
            "low": _to_float(row[4]),
            "volume": _to_float(row[5]) or 0,
            "amount": (
                _to_float(row[6])
                if len(row) > 6
                else None
            )
        })

    frame = pd.DataFrame(records)

    if frame.empty:
        raise RuntimeError(
            f"{code} 历史K线解析后为空"
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce"
    )

    for column in (
        "open", "close", "high", "low",
        "volume", "amount"
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce"
        )

    frame = frame.dropna(
        subset=["date", "open", "close", "high", "low"]
    )

    frame = frame.sort_values("date")
    frame = frame.drop_duplicates("date")
    frame = frame.reset_index(drop=True)

    minimum_rows = int(
        history_config.get("minimum_history_rows", 60)
    )

    if len(frame) < minimum_rows:
        raise RuntimeError(
            f"{code} 历史K线不足："
            f"{len(frame)} 行，至少需要 {minimum_rows} 行"
        )

    return frame


def get_data_source_status() -> dict:
    """返回本轮数据源状态。"""

    return json.loads(
        json.dumps(
            SOURCE_STATUS,
            ensure_ascii=False
        )
    )


def reset_data_source_status() -> None:
    """每轮扫描开始前清空状态。"""

    SOURCE_STATUS.update({
        "universe_source": None,
        "universe_count": 0,
        "snapshot_count": 0,
        "sina_count": 0,
        "tencent_count": 0,
        "errors": []
    })
