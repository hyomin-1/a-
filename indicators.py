
# -*- coding: utf-8 -*-
"""
A股猎手 - 技术指标模块

计算：
1. MA5、MA10、MA20、MA60
2. MACD
3. RSI
4. 成交量均线及量比
5. K线涨跌幅
6. 价格所处位置
7. 波动率
8. 技术风险提示

不负责选股评分，评分由 hunter.py 完成。
"""

import numpy as np
import pandas as pd


def _safe_float(value, default=None):
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _get_config(config):
    return config.get("indicators", {})


def calculate_indicators(
    history: pd.DataFrame,
    config: dict
) -> dict:
    """
    输入：
        history：腾讯历史K线DataFrame
        config：config.json配置

    输出：
        最新一日的技术指标字典
    """

    if history is None or history.empty:
        raise ValueError("历史K线为空，无法计算技术指标")

    df = history.copy()

    required = [
        "open",
        "close",
        "high",
        "low",
        "volume"
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"历史K线缺少字段：{column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "open",
            "close",
            "high",
            "low",
            "volume"
        ]
    )

    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 60:
        raise ValueError(
            f"历史K线不足60个交易日：当前{len(df)}日"
        )

    indicator_config = _get_config(config)

    close = df["close"]
    volume = df["volume"]

    # -------------------------
    # 一、移动平均线
    # -------------------------

    ma_periods = indicator_config.get(
        "ma_periods",
        [5, 10, 20, 60]
    )

    for period in ma_periods:
        period = int(period)

        df[f"ma{period}"] = close.rolling(
            window=period
        ).mean()

    # -------------------------
    # 二、MACD
    # -------------------------

    fast = int(
        indicator_config.get("macd_fast", 12)
    )

    slow = int(
        indicator_config.get("macd_slow", 26)
    )

    signal_period = int(
        indicator_config.get("macd_signal", 9)
    )

    ema_fast = close.ewm(
        span=fast,
        adjust=False
    ).mean()

    ema_slow = close.ewm(
        span=slow,
        adjust=False
    ).mean()

    df["dif"] = ema_fast - ema_slow

    df["dea"] = df["dif"].ewm(
        span=signal_period,
        adjust=False
    ).mean()

    df["macd"] = (
        df["dif"] - df["dea"]
    ) * 2

    # -------------------------
    # 三、RSI
    # -------------------------

    rsi_period = int(
        indicator_config.get("rsi_period", 14)
    )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / rsi_period,
        min_periods=rsi_period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / rsi_period,
        min_periods=rsi_period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # 没有下跌时，RSI按100处理。
    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "rsi"
    ] = 100

    # -------------------------
    # 四、成交量指标
    # -------------------------

    df["volume_ma5"] = volume.rolling(5).mean()
    df["volume_ma10"] = volume.rolling(10).mean()

    df["volume_ratio"] = (
        volume / df["volume_ma5"].replace(0, np.nan)
    )

    # -------------------------
    # 五、涨跌幅
    # -------------------------

    df["pct_change"] = close.pct_change() * 100

    df["pct_change_5"] = (
        close / close.shift(5) - 1
    ) * 100

    df["pct_change_10"] = (
        close / close.shift(10) - 1
    ) * 100

    # -------------------------
    # 六、价格位置
    # -------------------------

    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()

    df["high_60"] = df["high"].rolling(60).max()
    df["low_60"] = df["low"].rolling(60).min()

    df["position_20"] = (
        (close - df["low_20"])
        /
        (df["high_20"] - df["low_20"]).replace(0, np.nan)
    ) * 100

    df["position_60"] = (
        (close - df["low_60"])
        /
        (df["high_60"] - df["low_60"]).replace(0, np.nan)
    ) * 100

    # -------------------------
    # 七、波动率
    # -------------------------

    df["volatility_20"] = (
        df["pct_change"]
        .rolling(20)
        .std()
    )

    # -------------------------
    # 八、整理最新数据
    # -------------------------

    latest = df.iloc[-1]

    previous = df.iloc[-2]

    def value(column):
        return _safe_float(latest.get(column))

    def previous_value(column):
        return _safe_float(previous.get(column))

    result = {
        "date": str(latest["date"].date())
        if pd.notna(latest["date"])
        else None,

        "close": value("close"),

        "ma5": value("ma5"),
        "ma10": value("ma10"),
        "ma20": value("ma20"),
        "ma60": value("ma60"),

        "dif": value("dif"),
        "dea": value("dea"),
        "macd": value("macd"),

        "previous_macd": previous_value("macd"),

        "rsi": value("rsi"),

        "volume": value("volume"),
        "volume_ma5": value("volume_ma5"),
        "volume_ma10": value("volume_ma10"),
        "volume_ratio": value("volume_ratio"),

        "pct_change": value("pct_change"),
        "pct_change_5": value("pct_change_5"),
        "pct_change_10": value("pct_change_10"),

        "high_20": value("high_20"),
        "low_20": value("low_20"),
        "high_60": value("high_60"),
        "low_60": value("low_60"),

        "position_20": value("position_20"),
        "position_60": value("position_60"),

        "volatility_20": value("volatility_20"),

        "history_rows": int(len(df))
    }

    # -------------------------
    # 九、技术形态判断
    # -------------------------

    ma5 = result["ma5"]
    ma10 = result["ma10"]
    ma20 = result["ma20"]
    ma60 = result["ma60"]

    price = result["close"]

    if all(
        x is not None
        for x in [ma5, ma10, ma20, ma60, price]
    ):
        result["above_ma20"] = price >= ma20
        result["above_ma60"] = price >= ma60

        result["ma20_rising"] = (
            ma20 >= previous_value("ma20")
            if previous_value("ma20") is not None
            else False
        )

        result["ma60_rising"] = (
            ma60 >= previous_value("ma60")
            if previous_value("ma60") is not None
            else False
        )

        result["ma_bullish"] = (
            ma5 > ma10 > ma20
        )

        result["ma_full_bullish"] = (
            ma5 > ma10 > ma20 > ma60
        )

    else:
        result["above_ma20"] = False
        result["above_ma60"] = False
        result["ma20_rising"] = False
        result["ma60_rising"] = False
        result["ma_bullish"] = False
        result["ma_full_bullish"] = False

    dif = result["dif"]
    dea = result["dea"]
    macd = result["macd"]
    previous_macd = result["previous_macd"]

    result["macd_golden_cross"] = (
        dif is not None
        and dea is not None
        and dif > dea
        and (
            previous_value("dif") is not None
            and previous_value("dea") is not None
            and previous_value("dif") <= previous_value("dea")
        )
    )

    result["macd_positive"] = (
        macd is not None and macd > 0
    )

    result["macd_improving"] = (
        macd is not None
        and previous_macd is not None
        and macd > previous_macd
    )

    # -------------------------
    # 十、风险提示
    # -------------------------

    risk_flags = []

    rsi = result["rsi"]
    volume_ratio = result["volume_ratio"]
    position_20 = result["position_20"]

    if rsi is not None and rsi >= 75:
        risk_flags.append("RSI偏高，注意追高风险")

    if volume_ratio is not None and volume_ratio >= 4:
        risk_flags.append("成交量异常放大")

    if position_20 is not None and position_20 >= 95:
        risk_flags.append("接近20日区间高位")

    if (
        price is not None
        and ma20 is not None
        and price < ma20
        and ma20 is not None
    ):
        risk_flags.append("股价位于20日均线下方")

    if (
        macd is not None
        and previous_macd is not None
        and macd < previous_macd
    ):
        risk_flags.append("MACD动能减弱")

    result["risk_flags"] = risk_flags
    result["risk_count"] = len(risk_flags)

    return result


def calculate_indicators_batch(
    histories: dict,
    config: dict
) -> dict:
    """
    批量计算技术指标。

    histories格式：
    {
        "600000": DataFrame,
        "000001": DataFrame
    }
    """

    results = {}

    for code, history in histories.items():

        try:
            results[code] = calculate_indicators(
                history,
                config
            )

        except Exception as exc:
            results[code] = {
                "error": str(exc)
            }

    return results


# 兼容其他模块可能使用的函数名称。
def get_indicators(history, config):
    return calculate_indicators(history, config)
