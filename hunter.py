# -*- coding: utf-8 -*-
"""
A股猎手升级版（腾讯/新浪行情接口版）
15分钟MACD + 日线趋势 + 隔夜超短 + 历史复盘学习
"""

import os
import re
import json
import time
import traceback
import warnings
import requests

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TZ = ZoneInfo("Asia/Shanghai")
REPORT_DIR = "reports"
MAX_WORKERS = int(os.getenv("A_SHARE_WORKERS", "8"))
TOP_N = 3
MIN_SCORE = 0.0
HISTORY_DAYS = 120

TOP_INDUSTRY_BOARDS = 12
TOP_CONCEPT_BOARDS = 12


def load_config():
    path = os.getenv("ASHARE_CONFIG", "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


CONFIG = load_config()


def sh_now():
    return datetime.now(TZ)


def log(message):
    print(
        f"[{sh_now().strftime('%H:%M:%S')}] {message}",
        flush=True
    )


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

    gain = (
        delta.clip(lower=0)
        .rolling(period, min_periods=period)
        .mean()
    )

    loss = (
        (-delta.clip(upper=0))
        .rolling(period, min_periods=period)
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    result = result.mask(
        (loss == 0) & (gain > 0),
        100
    )

    return result


def calculate_indicators(df):
    """
    日线只作趋势背景：
    MA5/10/20/60 + 日线MACD + RSI。
    日线MACD不作为尾盘触发信号。
    """

    if df is None or df.empty:
        return None

    df = df.copy()

    close_col = get_col(
        df,
        ["close", "收盘"]
    )

    if not close_col:
        return None

    df[close_col] = pd.to_numeric(
        df[close_col],
        errors="coerce"
    )

    df = (
        df.dropna(subset=[close_col])
        .sort_index()
        .reset_index(drop=True)
    )

    if len(df) < 60:
        return None

    close = df[close_col]

    df["_close"] = close

    for n in (5, 10, 20, 60):
        df[f"MA{n}"] = close.rolling(n).mean()

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["DIF"] = ema12 - ema26

    df["DEA"] = df["DIF"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD"] = 2 * (
        df["DIF"] - df["DEA"]
    )

    df["RSI"] = calculate_rsi(
        close,
        14
    )

    high_col = get_col(
        df,
        ["high", "最高"]
    )

    low_col = get_col(
        df,
        ["low", "最低"]
    )

    open_col = get_col(
        df,
        ["open", "开盘"]
    )

    vol_col = get_col(
        df,
        ["volume", "成交量"]
    )

    if high_col:
        df["_high"] = pd.to_numeric(
            df[high_col],
            errors="coerce"
        )

    if low_col:
        df["_low"] = pd.to_numeric(
            df[low_col],
            errors="coerce"
        )

    if open_col:
        df["_open"] = pd.to_numeric(
            df[open_col],
            errors="coerce"
        )

    if vol_col:
        df["_volume"] = pd.to_numeric(
            df[vol_col],
            errors="coerce"
        )

    return df


def fetch_intraday_15m(code):
    """
    获取腾讯15分钟K线。

    使用腾讯 newfqkline 接口。
    失败时返回None，不用日线冒充分钟数据。
    """

    code = normalize_code(code)

    market = (
        "sh"
        if code.startswith(("6", "9"))
        else "sz"
    )

    symbol = f"{market}{code}"

    url = (
        "https://proxy.finance.qq.com/"
        "ifzqgtimg/appstock/app/newfqkline/get"
    )

    params = {
        "param": f"{symbol},m15,,,160,qfq"
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.qq.com/"
    }

    try:
        resp = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=12
        )

        resp.raise_for_status()

        payload = resp.json()

        node = (
            payload
            .get("data", {})
            .get(symbol, {})
        )

        rows = (
            node.get("m15")
            or node.get("qfq_m15")
            or []
        )

        if not rows:
            log(
                f"{code} 腾讯15分钟接口未返回K线"
            )
            return None

        # 腾讯分钟K线：
        # 时间、开盘、收盘、最高、最低、成交量

        df = pd.DataFrame(rows)

        if df.shape[1] < 6:
            log(
                f"{code} 腾讯15分钟K线字段不足"
            )
            return None

        df = df.iloc[:, :6]

        df.columns = [
            "时间",
            "开盘",
            "收盘",
            "最高",
            "最低",
            "成交量"
        ]

        for col in [
            "开盘",
            "收盘",
            "最高",
            "最低",
            "成交量"
        ]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = (
            df.dropna(subset=["收盘"])
            .reset_index(drop=True)
        )

        if len(df) < 30:
            log(
                f"{code} 腾讯15分钟K线不足30根，"
                f"实际{len(df)}根"
            )
            return None

        close = df["收盘"]

        dif = (
            close.ewm(
                span=12,
                adjust=False
            ).mean()
            -
            close.ewm(
                span=26,
                adjust=False
            ).mean()
        )

        dea = dif.ewm(
            span=9,
            adjust=False
        ).mean()

        df["DIF"] = dif
        df["DEA"] = dea

        df["MACD"] = 2 * (
            dif - dea
        )

        df["RSI15"] = calculate_rsi(
            close,
            14
        )

        return df

    except Exception as exc:
        log(
            f"{code} 腾讯15分钟数据不可用："
            f"{type(exc).__name__}: {exc}"
        )
        return None


def market_temperature(snapshot):
    """
    根据全市场行情快照计算市场热度。
    不把涨幅、成交额伪装成主力资金净流入。
    """

    df = snapshot.copy()

    pct = pd.to_numeric(
        df.get("涨跌幅"),
        errors="coerce"
    )

    amount = pd.to_numeric(
        df.get("成交额"),
        errors="coerce"
    )

    valid = pct.dropna()

    if valid.empty:
        return {
            "上涨家数": 0,
            "下跌家数": 0,
            "平盘家数": 0,
            "上涨比例": 0,
            "跌停附近": 0,
            "涨停附近": 0,
            "成交额亿元": None,
            "热度": "数据不足"
        }

    up = int((valid > 0).sum())
    down = int((valid < 0).sum())
    flat = int((valid == 0).sum())

    up_limit = int(
        (valid >= 9.5).sum()
    )

    down_limit = int(
        (valid <= -9.5).sum()
    )

    breadth = up / max(1, len(valid))

    total_amount = float(
        amount.fillna(0).sum() / 1e8
    )

    if (
        breadth >= 0.65
        and up_limit >= 50
        and down_limit <= 10
    ):
        mood = "偏热"

    elif (
        breadth >= 0.52
        and down > 0
        and down_limit <= 20
    ):
        mood = "温和偏强"

    elif (
        breadth <= 0.35
        or down_limit >= 30
    ):
        mood = "偏冷"

    else:
        mood = "分化"

    return {
        "上涨家数": up,
        "下跌家数": down,
        "平盘家数": flat,
        "上涨比例": round(
            breadth * 100,
            1
        ),
        "涨停附近": up_limit,
        "跌停附近": down_limit,
        "成交额亿元": round(
            total_amount,
            1
        ),
        "热度": mood
    }


def build_sector_heat(snapshot):
    """
    如果行情快照含行业字段，
    按行业平均涨幅、上涨比例、成交额计算板块热度。

    当前接口没有可靠的行业主力净流入字段，
    因此明确标注资金净流入不可验证。
    """

    if "行业" not in snapshot.columns:
        return (
            pd.DataFrame(),
            "行情快照未提供行业字段，板块热度不可计算"
        )

    df = snapshot.copy()

    df["涨跌幅"] = pd.to_numeric(
        df.get("涨跌幅"),
        errors="coerce"
    )

    df["成交额"] = pd.to_numeric(
        df.get("成交额"),
        errors="coerce"
    )

    df = df.dropna(
        subset=["行业", "涨跌幅"]
    )

    if df.empty:
        return (
            pd.DataFrame(),
            "行业字段无有效数据"
        )

    grouped = df.groupby("行业").agg(
        股票数=("代码", "count"),
        平均涨幅=("涨跌幅", "mean"),
        上涨家数=(
            "涨跌幅",
            lambda x: int((x > 0).sum())
        ),
        成交额亿元=(
            "成交额",
            lambda x: float(
                x.fillna(0).sum() / 1e8
            )
        )
    ).reset_index()

    grouped["上涨比例"] = (
        grouped["上涨家数"]
        / grouped["股票数"]
        * 100
    )

    grouped = grouped.sort_values(
        [
            "平均涨幅",
            "上涨比例",
            "成交额亿元"
        ],
        ascending=False
    ).head(15)

    grouped["资金净流入"] = (
        "当前数据源未提供，不能确认"
    )

    return (
        grouped,
        "板块热度为行业涨幅/广度/成交额代理，"
        "不等于主力资金净流入"
    )


def fetch_board_flow():
    status = {
        "行业资金净流入": "当前数据源未提供，不能确认",
        "概念资金净流入": "当前数据源未提供，不能确认"
    }

    boards = pd.DataFrame(
        columns=[
            "板块类型",
            "板块名称",
            "资金流排名",
            "资金流值",
            "板块涨幅%"
        ]
    )

    return boards, {}, status


def load_previous_picks(today):
    ensure_dirs()

    candidates = []

    run_dir = os.path.join(
        REPORT_DIR,
        "runs"
    )

    if os.path.isdir(run_dir):
        for filename in os.listdir(run_dir):

            match = re.fullmatch(
                r"(\d{4}-\d{2}-\d{2})_(\d{6})\.csv",
                filename
            )

            if match and match.group(1) < today:
                candidates.append(
                    (
                        match.group(1) + " " + match.group(2),
                        os.path.join(run_dir, filename)
                    )
                )

    if candidates:
        stamp, path = sorted(
            candidates,
            reverse=True
        )[0]

        try:
            df = pd.read_csv(
                path,
                dtype={"代码": str}
            )

            if (
                not df.empty
                and "代码" in df.columns
            ):
                df["代码"] = (
                    df["代码"].map(normalize_code)
                )

            return stamp, df

        except Exception as exc:
            log(
                f"读取上次运行快照失败：{exc}"
            )

    daily = []

    for filename in os.listdir(REPORT_DIR):

        if re.fullmatch(
            r"\d{4}-\d{2}-\d{2}\.csv",
            filename
        ):
            date_text = filename[:-4]

            if date_text < today:
                daily.append(
                    (
                        date_text,
                        os.path.join(
                            REPORT_DIR,
                            filename
                        )
                    )
                )

    if not daily:
        return None, pd.DataFrame()

    prev_date, path = sorted(
        daily,
        reverse=True
    )[0]

    try:
        df = pd.read_csv(
            path,
            dtype={"代码": str}
        )

        if (
            not df.empty
            and "代码" in df.columns
        ):
            df["代码"] = (
                df["代码"].map(normalize_code)
            )

        return prev_date, df

    except Exception as exc:
        log(
            f"读取上次推荐文件失败：{exc}"
        )
        return prev_date, pd.DataFrame()


def learning_paths():
    return {
        "outcomes": os.path.join(
            REPORT_DIR,
            "learning_outcomes.csv"
        ),
        "memory": os.path.join(
            REPORT_DIR,
            "learning_memory.json"
        ),
        "rules": os.path.join(
            REPORT_DIR,
            "learned_rules.json"
        )
    }


def read_json(path, default):
    try:
        if os.path.exists(path):
            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:
                return json.load(f)

    except Exception as exc:
        log(
            f"读取学习文件失败 {path}: {exc}"
        )

    return default
