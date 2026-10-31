
# -*- coding: utf-8 -*-
"""
A股猎手 - 长期学习与历史复盘模块

功能：
1. 保存每次扫描的推荐股票
2. 持久化保存历史推荐记录
3. 跟踪推荐股票第1、5、10个交易日的表现
4. 分析历史亏损样本
5. 根据历史表现调整相似股票评分
6. 不因单次亏损就永久惩罚某种形态

数据文件：
reports/learning_memory.json
reports/recommendation_history.json
reports/review_history.json

说明：
历史表现只用于辅助调整评分，不代表未来收益。
"""

import os
import json
import math
import logging
from datetime import datetime

import pandas as pd

from data_source import get_history


logger = logging.getLogger("ashare_learning")


MEMORY_FILE = "learning_memory.json"
RECOMMENDATION_FILE = "recommendation_history.json"
REVIEW_FILE = "review_history.json"


# ============================================================
# 一、基础工具
# ============================================================

def _safe_float(value, default=None):
    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _load_json(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as exc:
        logger.warning(
            "读取学习文件失败：%s，原因：%s",
            path,
            exc
        )

        return default


def _save_json(path, data):
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True
    )

    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    os.replace(temp_path, path)


def _paths(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    return {
        "memory": os.path.join(
            output_dir,
            MEMORY_FILE
        ),
        "recommendations": os.path.join(
            output_dir,
            RECOMMENDATION_FILE
        ),
        "reviews": os.path.join(
            output_dir,
            REVIEW_FILE
        )
    }


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _normalize_code(code):
    return str(code).strip().replace("sh", "").replace("sz", "")


# ============================================================
# 二、初始化长期记忆
# ============================================================

def load_learning_memory(output_dir="reports"):
    """
    读取长期学习记录。

    如果没有历史文件，初始化空记录。
    """

    paths = _paths(output_dir)

    memory = _load_json(
        paths["memory"],
        {
            "version": 1,
            "created_at": _today(),
            "updated_at": _today(),
            "total_recommendations": 0,
            "total_reviews": 0,
            "patterns": {},
            "statistics": {
                "reviewed_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "average_return_1d": None,
                "average_return_5d": None,
                "average_return_10d": None
            }
        }
    )

    if not isinstance(memory, dict):
        raise ValueError("学习记忆文件格式异常")

    memory.setdefault("patterns", {})
    memory.setdefault("statistics", {})

    return memory


def _load_recommendations(output_dir):
    paths = _paths(output_dir)

    records = _load_json(
        paths["recommendations"],
        []
    )

    if not isinstance(records, list):
        return []

    return records


def _load_reviews(output_dir):
    paths = _paths(output_dir)

    records = _load_json(
        paths["reviews"],
        []
    )

    if not isinstance(records, list):
        return []

    return records


# ============================================================
# 三、识别股票形态
# ============================================================

def classify_stock_pattern(stock, indicators):
    """
    将股票分组，供历史样本比较。

    分组不是预测模型，只是历史表现统计标签。
    """

    position = _safe_float(
        indicators.get("position_60")
    )

    rsi = _safe_float(
        indicators.get("rsi")
    )

    volume_ratio = _safe_float(
        indicators.get("volume_ratio")
    )

    pct = _safe_float(
        stock.get("change_pct"),
        0
    )

    above_ma20 = bool(
        indicators.get("above_ma20")
    )

    macd_positive = bool(
        indicators.get("macd_positive")
    )

    if position is None:
        position_group = "位置未知"

    elif position < 30:
        position_group = "60日低位"

    elif position < 60:
        position_group = "60日中低位"

    elif position < 85:
        position_group = "60日中高位"

    else:
        position_group = "60日高位"

    if rsi is None:
        rsi_group = "RSI未知"

    elif rsi < 35:
        rsi_group = "RSI偏弱"

    elif rsi < 50:
        rsi_group = "RSI中低"

    elif rsi < 65:
        rsi_group = "RSI正常"

    else:
        rsi_group = "RSI偏热"

    if volume_ratio is None:
        volume_group = "量比未知"

    elif volume_ratio < 1:
        volume_group = "缩量"

    elif volume_ratio < 2:
        volume_group = "温和放量"

    elif volume_ratio < 3:
        volume_group = "明显放量"

    else:
        volume_group = "异常放量"

    if pct < 0:
        change_group = "当日下跌"

    elif pct <= 2:
        change_group = "小幅上涨"

    elif pct <= 4:
        change_group = "温和上涨"

    else:
        change_group = "较大涨幅"

    trend_group = (
        "站上MA20"
        if above_ma20
        else "低于MA20"
    )

    macd_group = (
        "MACD偏强"
        if macd_positive
        else "MACD偏弱"
    )

    return "|".join([
        position_group,
        rsi_group,
        volume_group,
        change_group,
        trend_group,
        macd_group
    ])


# ============================================================
# 四、保存本轮推荐
# ============================================================

def record_scan(
    output_dir,
    run_id,
    recommendations,
    market_summary
):
    """
    保存本轮推荐股票。

    每条记录保存：
    - 推荐日期
    - 股票代码
    - 股票名称
    - 推荐价格
    - 综合评分
    - 技术指标
    - 形态分类
    - 待复盘交易日
    """

    paths = _paths(output_dir)

    records = _load_json(
        paths["recommendations"],
        []
    )

    if not isinstance(records, list):
        records = []

    existing_ids = {
        item.get("record_id")
        for item in records
        if isinstance(item, dict)
    }

    new_records = []

    for stock in recommendations:

        code = _normalize_code(
            stock.get("code", "")
        )

        if not code:
            continue

        indicators = stock.get(
            "indicators",
            {}
        )

        record_id = f"{run_id}_{code}"

        if record_id in existing_ids:
            continue

        price = _safe_float(
            stock.get("price")
        )

        if price is None or price <= 0:
            continue

        pattern = classify_stock_pattern(
            stock,
            indicators
        )

        record = {
            "record_id": record_id,
            "run_id": run_id,
            "date": _today(),
            "code": code,
            "name": stock.get("name", ""),
            "recommend_price": price,
            "recommend_score": _safe_float(
                stock.get("score"),
                0
            ),
            "change_pct": _safe_float(
                stock.get("change_pct"),
                0
            ),
            "pattern": pattern,
            "indicators": {
                "rsi": _safe_float(
                    indicators.get("rsi")
                ),
                "macd": _safe_float(
                    indicators.get("macd")
                ),
                "volume_ratio": _safe_float(
                    indicators.get("volume_ratio")
                ),
                "position_20": _safe_float(
                    indicators.get("position_20")
                ),
                "position_60": _safe_float(
                    indicators.get("position_60")
                ),
                "above_ma20": bool(
                    indicators.get("above_ma20")
                ),
                "macd_positive": bool(
                    indicators.get("macd_positive")
                )
            },
            "market_environment": market_summary.get(
                "市场环境"
            ),
            "reviewed": {},
            "review_status": "待复盘"
        }

        new_records.append(record)

    records.extend(new_records)

    _save_json(
        paths["recommendations"],
        records
    )

    memory = load_learning_memory(output_dir)

    memory["total_recommendations"] = len(records)
    memory["updated_at"] = _today()

    _save_json(
        paths["memory"],
        memory
    )

    logger.info(
        "本轮新增学习记录：%s 条",
        len(new_records)
    )

    return new_records


# ============================================================
# 五、计算历史收益
# ============================================================

def _get_future_return(history, recommend_price, sessions):
    """
    根据推荐价格计算未来第N个交易日收益率。

    推荐日为第0日。
    第1个交易日是推荐日期之后的第一根K线。
    """

    if history is None or history.empty:
        return None

    if len(history) <= sessions:
        return None

    future_row = history.iloc[sessions]

    future_price = _safe_float(
        future_row.get("close")
    )

    if future_price is None or recommend_price <= 0:
        return None

    return round(
        (future_price / recommend_price - 1) * 100,
        4
    )


def _get_review_result(return_pct):
    if return_pct is None:
        return "数据不足"

    if return_pct >= 3:
        return "表现较好"

    if return_pct > 0:
        return "小幅盈利"

    if return_pct > -2:
        return "小幅亏损"

    return "表现较差"


# ============================================================
# 六、复盘历史推荐
# ============================================================

def review_pending_recommendations(
    output_dir,
    config
):
    """
    检查历史推荐的第1、5、10个交易日表现。

    只有目标交易日已经出现时才更新记录。
    不会用当前价格冒充未来收益。
    """

    paths = _paths(output_dir)

    records = _load_recommendations(output_dir)

    if not records:
        return {
            "reviewed": 0,
            "updated": 0
        }

    review_sessions = config.get(
        "learning",
        {}
    ).get(
        "review_after_sessions",
        [1, 5, 10]
    )

    review_sessions = [
        int(x) for x in review_sessions
    ]

    updated_count = 0
    reviewed_count = 0

    for record in records:

        code = record.get("code")

        if not code:
            continue

        recommend_price = _safe_float(
            record.get("recommend_price")
        )

        if recommend_price is None:
            continue

        if all(
            str(session) in record.get("reviewed", {})
            for session in review_sessions
        ):
            record["review_status"] = "复盘完成"
            continue

        try:
            history = get_history(
                code,
                config
            )

        except Exception as exc:
            logger.warning(
                "复盘获取历史行情失败：%s，原因：%s",
                code,
                exc
            )
            continue

        # 找到推荐日期在K线中的位置。
        recommend_date = record.get("date")

        if not recommend_date:
            continue

        history_dates = pd.to_datetime(
            history["date"]
        ).dt.strftime("%Y-%m-%d")

        positions = [
            index
            for index, date in enumerate(history_dates)
            if date == recommend_date
        ]

        if positions:
            start_index = positions[-1]

        else:
            # 如果推荐发生在当天收盘前，腾讯日K可能尚未包含当天。
            # 此时不能直接把最新K线当作推荐日。
            # 尝试使用推荐日期之后的第一根K线作为参考起点。
            future_positions = [
                index
                for index, date in enumerate(history_dates)
                if date > recommend_date
            ]

            if not future_positions:
                continue

            start_index = future_positions[0] - 1

        for session in review_sessions:

            key = str(session)

            if key in record.get("reviewed", {}):
                continue

            target_index = start_index + session

            if target_index >= len(history):
                continue

            target_price = _safe_float(
                history.iloc[target_index].get("close")
            )

            if target_price is None:
                continue

            return_pct = round(
                (target_price / recommend_price - 1) * 100,
                4
            )

            review = {
                "sessions": session,
                "date": str(
                    pd.to_datetime(
                        history.iloc[target_index]["date"]
                    ).date()
                ),
                "price": target_price,
                "return_pct": return_pct,
                "result": _get_review_result(return_pct)
            }

            record.setdefault(
                "reviewed",
                {}
            )[key] = review

            reviewed_count += 1
            updated_count += 1

        if all(
            str(session) in record.get("reviewed", {})
            for session in review_sessions
        ):
            record["review_status"] = "复盘完成"

        elif record.get("reviewed"):
            record["review_status"] = "部分完成"

    _save_json(
        paths["recommendations"],
        records
    )

    # 同时保留独立复盘历史。
    review_records = []

    for record in records:
        for session, result in record.get(
            "reviewed",
            {}
        ).items():

            review_records.append({
                "record_id": record.get("record_id"),
                "code": record.get("code"),
                "name": record.get("name"),
                "pattern": record.get("pattern"),
                "sessions": int(session),
                "return_pct": result.get("return_pct"),
                "result": result.get("result"),
                "date": result.get("date")
            })

    _save_json(
        paths["reviews"],
        review_records
    )

    update_learning_statistics(
        output_dir
    )

    return {
        "reviewed": reviewed_count,
        "updated": updated_count,
        "total_records": len(records)
    }


# ============================================================
# 七、更新长期统计
# ============================================================

def update_learning_statistics(output_dir):
    records = _load_recommendations(output_dir)

    memory = load_learning_memory(output_dir)

    statistics = {
        "reviewed_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "average_return_1d": None,
        "average_return_5d": None,
        "average_return_10d": None
    }

    for session in (1, 5, 10):

        values = []

        for record in records:

            review = record.get(
                "reviewed",
                {}
            ).get(str(session))

            if not review:
                continue

            value = _safe_float(
                review.get("return_pct")
            )

            if value is not None:
                values.append(value)

        if values:

            statistics[
                f"average_return_{session}d"
            ] = round(
                sum(values) / len(values),
                4
            )

            if session == 5:
                statistics["reviewed_count"] = len(values)

                statistics["positive_count"] = sum(
                    1 for value in values
                    if value > 0
                )

                statistics["negative_count"] = sum(
                    1 for value in values
                    if value < 0
                )

    memory["statistics"] = statistics

    memory["total_recommendations"] = len(records)

    memory["total_reviews"] = sum(
        len(record.get("reviewed", {}))
        for record in records
    )

    memory["updated_at"] = _today()

    # 按形态统计第5个交易日表现。
    patterns = {}

    for record in records:

        pattern = record.get("pattern", "未知")

        review = record.get(
            "reviewed",
            {}
        ).get("5")

        if not review:
            continue

        return_pct = _safe_float(
            review.get("return_pct")
        )

        if return_pct is None:
            continue

        patterns.setdefault(
            pattern,
            []
        ).append(return_pct)

    pattern_summary = {}

    for pattern, values in patterns.items():

        pattern_summary[pattern] = {
            "samples": len(values),
            "average_return_5d": round(
                sum(values) / len(values),
                4
            ),
            "negative_ratio": round(
                sum(1 for value in values if value < 0)
                / len(values),
                4
            )
        }

    memory["patterns"] = pattern_summary

    paths = _paths(output_dir)

    _save_json(
        paths["memory"],
        memory
    )

    return memory


# ============================================================
# 八、根据历史表现调整评分
# ============================================================

def get_learning_adjustment(
    stock,
    indicators,
    memory,
    config
):
    """
    根据历史相似形态的表现调整评分。

    规则：
    1. 样本不足不调整。
    2. 样本日期不足不调整。
    3. 平均收益较差时适度扣分。
    4. 平均收益较好时最多小幅加分。
    5. 单个形态的调整幅度有限。

    返回范围：
    -1.0 到 +0.3 分。
    """

    learning_config = config.get(
        "learning",
        {}
    )

    if not learning_config.get(
        "apply_learning",
        True
    ):
        return 0.0

    pattern = classify_stock_pattern(
        stock,
        indicators
    )

    pattern_data = memory.get(
        "patterns",
        {}
    ).get(pattern)

    if not pattern_data:
        return 0.0

    samples = int(
        pattern_data.get("samples", 0)
    )

    minimum_samples = int(
        learning_config.get(
            "min_learning_group",
            12
        )
    )

    if samples < minimum_samples:
        return 0.0

    average_return = _safe_float(
        pattern_data.get("average_return_5d")
    )

    negative_ratio = _safe_float(
        pattern_data.get("negative_ratio")
    )

    if average_return is None or negative_ratio is None:
        return 0.0

    adjustment = 0.0

    if average_return <= -3 and negative_ratio >= 0.65:
        adjustment = -1.0

    elif average_return <= -1.5 and negative_ratio >= 0.60:
        adjustment = -0.6

    elif average_return < 0 and negative_ratio >= 0.55:
        adjustment = -0.3

    elif average_return >= 3 and negative_ratio <= 0.35:
        adjustment = 0.3

    elif average_return >= 1.5 and negative_ratio <= 0.40:
        adjustment = 0.15

    max_penalty = float(
        learning_config.get(
            "max_learning_penalty",
            1.0
        )
    )

    return max(
        -max_penalty,
        min(0.3, adjustment)
    )


# ============================================================
# 九、完整复盘入口
# ============================================================

def run_learning_review(
    output_dir,
    config
):
    """
    单独调用长期学习复盘。

    hunter.py 后续每次扫描都可以调用。
    """

    result = review_pending_recommendations(
        output_dir,
        config
    )

    memory = update_learning_statistics(
        output_dir
    )

    return {
        "review_result": result,
        "statistics": memory.get("statistics", {}),
        "pattern_count": len(
            memory.get("patterns", {})
        )
    }
