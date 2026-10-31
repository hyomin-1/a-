
# -*- coding: utf-8 -*-
"""
A股猎手 - 核心选股引擎

主要功能：
1. 获取沪深A股全市场行情
2. 计算市场整体涨跌情况
3. 初步筛选成交活跃、价格合理的股票
4. 获取候选股票历史K线
5. 计算技术指标
6. 进行10分制综合评分
7. 输出评分最高的3只股票
8. 保存扫描报告和数据源状态
9. 调用 learning.py 保存推荐记录和学习数据

数据源：
- 实时行情：新浪优先，腾讯备用
- 历史K线：腾讯
- 不使用东方财富和 AkShare
"""

import os
import sys
import json
import csv
import time
import argparse
import logging
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from data_source import (
    get_market_snapshot,
    get_history,
    get_data_source_status,
    reset_data_source_status
)

from indicators import calculate_indicators


# ============================================================
# 一、基础设置
# ============================================================

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("ashare_hunter")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        os.path.join(output_dir, "hunter.log"),
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    return config


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

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


def safe_float(value, default=None):
    try:
        result = float(value)

        if pd.isna(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def now_text():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 二、市场环境分析
# ============================================================

def calculate_market_summary(snapshot):
    """
    统计全市场涨跌情况。

    注意：
    这里是市场宽度，不代表板块主力资金净流入。
    """

    if not snapshot:
        raise RuntimeError("市场快照为空，无法计算市场环境")

    total = len(snapshot)

    up = 0
    down = 0
    flat = 0
    limit_up = 0
    limit_down = 0

    valid_amount = []

    for item in snapshot:

        pct = safe_float(item.get("change_pct"), 0)
        amount = safe_float(item.get("amount"), 0)

        if pct > 0:
            up += 1
        elif pct < 0:
            down += 1
        else:
            flat += 1

        if pct >= 9.7:
            limit_up += 1

        if pct <= -9.7:
            limit_down += 1

        if amount > 0:
            valid_amount.append(amount)

    breadth = up / total if total else 0

    avg_change = (
        sum(
            safe_float(item.get("change_pct"), 0)
            for item in snapshot
        ) / total
        if total else 0
    )

    total_amount = sum(valid_amount)

    if breadth >= 0.65:
        environment = "市场普涨"
    elif breadth >= 0.50:
        environment = "市场偏强"
    elif breadth >= 0.35:
        environment = "市场分化"
    else:
        environment = "市场偏弱"

    return {
        "total_stocks": total,
        "上涨家数": up,
        "下跌家数": down,
        "平盘家数": flat,
        "涨停附近家数": limit_up,
        "跌停附近家数": limit_down,
        "上涨比例": round(breadth * 100, 2),
        "平均涨跌幅": round(avg_change, 3),
        "市场成交额": round(total_amount, 2),
        "市场环境": environment,
        "sector_fund_flow_available": False,
        "sector_strength_available": False,
        "说明": (
            "当前数据源未提供可靠的全行业主力资金流向及完整行业分类；"
            "本报告使用全市场涨跌家数和成交活跃度作为市场环境参考，"
            "不将其冒充板块资金净流入。"
        )
    }


# ============================================================
# 三、初步筛选
# ============================================================

def preliminary_screen(snapshot, config):
    """
    从全市场快照中筛选候选股票。

    先按价格、涨幅、成交额过滤，
    再按成交额排序取前N只进行历史K线分析。
    """

    settings = config.get("screening", {})

    min_price = float(settings.get("min_price", 3))
    max_price = float(settings.get("max_price", 150))

    min_amount = float(
        settings.get("min_amount_yuan", 100000000)
    )

    min_pct = float(settings.get("min_day_pct", -1))
    max_pct = float(settings.get("max_day_pct", 5.5))

    shortlist_size = int(
        settings.get("shortlist_size", 60)
    )

    candidates = []

    for item in snapshot:

        price = safe_float(item.get("price"))
        amount = safe_float(item.get("amount"), 0)
        pct = safe_float(item.get("change_pct"))

        if price is None or pct is None:
            continue

        if price < min_price or price > max_price:
            continue

        if amount < min_amount:
            continue

        if pct < min_pct or pct > max_pct:
            continue

        candidates.append(item)

    candidates.sort(
        key=lambda x: safe_float(x.get("amount"), 0),
        reverse=True
    )

    return candidates[:shortlist_size]


# ============================================================
# 四、综合评分
# ============================================================

def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def score_market_environment(market_summary):
    """
    市场环境分值，满分1分。
    """

    breadth = (
        safe_float(market_summary.get("上涨比例"), 0) / 100
    )

    if breadth >= 0.70:
        return 1.0

    if breadth >= 0.60:
        return 0.9

    if breadth >= 0.50:
        return 0.75

    if breadth >= 0.40:
        return 0.6

    if breadth >= 0.30:
        return 0.4

    return 0.2


def score_low_position(indicators):
    """
    低位评分，满分1分。

    主要参考：
    - 20日区间位置
    - 60日区间位置
    - 是否远离长期均线
    """

    position_20 = safe_float(
        indicators.get("position_20")
    )

    position_60 = safe_float(
        indicators.get("position_60")
    )

    if position_20 is None or position_60 is None:
        return 0.0

    score_20 = 1 - clamp(position_20 / 100)
    score_60 = 1 - clamp(position_60 / 100)

    return (
        score_20 * 0.55
        + score_60 * 0.45
    )


def score_technical(indicators):
    """
    技术趋势评分，满分2.5分。
    """

    score = 0.0

    if indicators.get("above_ma20"):
        score += 0.35

    if indicators.get("above_ma60"):
        score += 0.35

    if indicators.get("ma20_rising"):
        score += 0.30

    if indicators.get("ma_bullish"):
        score += 0.35

    if indicators.get("macd_positive"):
        score += 0.35

    if indicators.get("macd_improving"):
        score += 0.30

    if indicators.get("macd_golden_cross"):
        score += 0.25

    rsi = safe_float(indicators.get("rsi"))

    if rsi is not None:
        if 45 <= rsi <= 65:
            score += 0.25
        elif 35 <= rsi < 45:
            score += 0.15
        elif 65 < rsi <= 70:
            score += 0.10

    return min(score, 2.5)


def score_price_volume(stock, indicators):
    """
    量价评分，满分1.5分。
    """

    score = 0.0

    pct = safe_float(stock.get("change_pct"), 0)
    volume_ratio = safe_float(
        indicators.get("volume_ratio")
    )

    if -0.5 <= pct <= 3:
        score += 0.45
    elif -1 <= pct < -0.5:
        score += 0.25
    elif 3 < pct <= 4.5:
        score += 0.25

    if volume_ratio is not None:

        if 1.15 <= volume_ratio <= 2.2:
            score += 0.75

        elif 1.0 <= volume_ratio < 1.15:
            score += 0.45

        elif 2.2 < volume_ratio <= 2.8:
            score += 0.45

    amount = safe_float(stock.get("amount"), 0)

    if amount >= 500000000:
        score += 0.30

    elif amount >= 200000000:
        score += 0.20

    return min(score, 1.5)


def score_risk(indicators, config):
    """
    风险扣分。
    """

    risk_config = config.get("risk", {})

    penalty = 0.0

    rsi = safe_float(indicators.get("rsi"))
    volume_ratio = safe_float(
        indicators.get("volume_ratio")
    )

    position_20 = safe_float(
        indicators.get("position_20")
    )

    if rsi is not None:
        if rsi >= float(risk_config.get("avoid_high_rsi", 75)):
            penalty += 0.8

    if volume_ratio is not None:
        if volume_ratio >= float(
            risk_config.get("avoid_extreme_volume_ratio", 4)
        ):
            penalty += 0.8

    if position_20 is not None and position_20 >= 98:
        penalty += 0.5

    if indicators.get("risk_count", 0) >= 3:
        penalty += 0.3

    return min(penalty, 2.0)


def calculate_stock_score(
    stock,
    indicators,
    market_summary,
    config,
    learning_adjustment=0.0
):
    """
    10分制综合评分。

    权重：
    技术趋势：2.5
    量价：1.5
    低位：1.5
    市场环境：1.0
    板块强度：暂不可用，不伪造
    委比/内外盘：暂不可用，不伪造

    不可用项目不参与评分归一化。
    """

    weights = config.get("scoring", {})

    technical_weight = float(
        weights.get("technical_weight", 2.5)
    )

    price_volume_weight = float(
        weights.get("price_volume_weight", 1.5)
    )

    low_position_weight = float(
        weights.get("low_position_weight", 1.5)
    )

    market_weight = float(
        weights.get("market_environment_weight", 1.0)
    )

    technical = score_technical(indicators) / 2.5
    price_volume = score_price_volume(
        stock,
        indicators
    ) / 1.5

    low_position = score_low_position(indicators)

    market = score_market_environment(market_summary)

    components = [
        (technical, technical_weight),
        (price_volume, price_volume_weight),
        (low_position, low_position_weight),
        (market, market_weight)
    ]

    available_weight = sum(
        weight for _, weight in components
    )

    weighted_score = sum(
        value * weight
        for value, weight in components
    )

    base_score = (
        weighted_score / available_weight * 10
        if available_weight > 0
        else 0
    )

    penalty = score_risk(indicators, config)

    adjustment = safe_float(
        learning_adjustment,
        0
    )

    max_learning_penalty = float(
        config.get("learning", {}).get(
            "max_learning_penalty",
            1.0
        )
    )

    adjustment = max(
        -max_learning_penalty,
        min(max_learning_penalty, adjustment)
    )

    final_score = max(
        0,
        min(10, base_score - penalty + adjustment)
    )

    return {
        "score": round(final_score, 2),
        "base_score": round(base_score, 2),
        "risk_penalty": round(penalty, 2),
        "learning_adjustment": round(adjustment, 2),
        "technical_score": round(technical * 2.5, 2),
        "price_volume_score": round(price_volume * 1.5, 2),
        "low_position_score": round(low_position * 1.5, 2),
        "market_score": round(market, 2),
        "sector_score": None,
        "order_book_score": None,
        "score_note": (
            "当前评分未包含可靠的板块主力资金、委比和内外盘数据；"
            "不可用指标不以虚构数值参与评分。"
        )
    }


# ============================================================
# 五、单只股票分析
# ============================================================

def analyze_one_stock(
    stock,
    config,
    market_summary,
    learning_memory=None
):
    code = stock["code"]

    history = get_history(code, config)

    indicators = calculate_indicators(
        history,
        config
    )

    learning_adjustment = 0.0

    # learning.py 在下一份文件中提供该接口。
    if learning_memory is not None:
        try:
            from learning import get_learning_adjustment

            learning_adjustment = get_learning_adjustment(
                stock,
                indicators,
                learning_memory,
                config
            )

        except Exception:
            learning_adjustment = 0.0

    score_result = calculate_stock_score(
        stock,
        indicators,
        market_summary,
        config,
        learning_adjustment
    )

    result = dict(stock)

    result["indicators"] = indicators
    result.update(score_result)

    result["risk_flags"] = indicators.get(
        "risk_flags",
        []
    )

    result["sector"] = "行业分类数据暂不可用"

    result["sector_fund_flow"] = None
    result["委比"] = None
    result["内盘"] = None
    result["外盘"] = None

    result["参考止盈幅度"] = config.get(
        "risk", {}
    ).get("reference_take_profit_pct", 3.0)

    result["参考止损幅度"] = config.get(
        "risk", {}
    ).get("reference_stop_loss_pct", 2.0)

    result["说明"] = (
        "止盈止损幅度仅为风险管理参考，不是收益保证，"
        "实际交易还需考虑价格跳空、滑点和市场波动。"
    )

    return result


# ============================================================
# 六、候选股票批量分析
# ============================================================

def analyze_candidates(
    candidates,
    config,
    market_summary,
    logger
):
    results = []

    learning_memory = None

    if config.get("learning", {}).get("enabled", True):
        try:
            from learning import load_learning_memory

            learning_memory = load_learning_memory(
                config.get("output", {}).get(
                    "directory",
                    "reports"
                )
            )

        except Exception as exc:
            logger.warning(
                "学习记录暂时无法读取，本轮按基础模型评分：%s",
                exc
            )

    workers = int(
        config.get("data_source", {}).get(
            "max_workers",
            4
        )
    )

    workers = max(1, min(workers, 6))

    with ThreadPoolExecutor(max_workers=workers) as executor:

        future_map = {
            executor.submit(
                analyze_one_stock,
                stock,
                config,
                market_summary,
                learning_memory
            ): stock
            for stock in candidates
        }

        for future in as_completed(future_map):

            stock = future_map[future]

            try:
                result = future.result()
                results.append(result)

                logger.info(
                    "完成分析：%s %s，评分 %.2f",
                    result.get("code"),
                    result.get("name"),
                    result.get("score", 0)
                )

            except Exception as exc:
                logger.warning(
                    "股票分析失败：%s %s，原因：%s",
                    stock.get("code"),
                    stock.get("name"),
                    exc
                )

    results.sort(
        key=lambda item: item.get("score", 0),
        reverse=True
    )

    return results


# ============================================================
# 七、报告输出
# ============================================================

def save_candidates_csv(path, results):
    fields = [
        "code",
        "name",
        "price",
        "change_pct",
        "amount",
        "score",
        "base_score",
        "risk_penalty",
        "learning_adjustment",
        "technical_score",
        "price_volume_score",
        "low_position_score",
        "market_score",
        "sector",
        "参考止盈幅度",
        "参考止损幅度"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()

        for item in results:
            writer.writerow(item)


def save_markdown_report(
    path,
    run_id,
    market_summary,
    candidates,
    recommendations,
    source_status
):
    lines = []

    lines.append("# A股猎手扫描报告")
    lines.append("")
    lines.append(f"- 扫描编号：{run_id}")
    lines.append(f"- 扫描时间：{now_text()}")
    lines.append("- 行情主源：新浪")
    lines.append("- 行情备用：腾讯")
    lines.append("- 历史K线：腾讯")
    lines.append("")

    lines.append("## 一、市场环境")
    lines.append("")

    for key, value in market_summary.items():
        lines.append(f"- **{key}**：{value}")

    lines.append("")

    lines.append("## 二、综合评分前三名")
    lines.append("")

    if not recommendations:
        lines.append("本轮没有符合条件且成功完成分析的股票。")
    else:
        for index, item in enumerate(recommendations, 1):

            indicators = item.get("indicators", {})

            lines.append(
                f"### {index}. {item.get('name')} "
                f"({item.get('code')})"
            )

            lines.append(
                f"- 综合评分：**{item.get('score')} / 10**"
            )

            lines.append(
                f"- 最新价格：{item.get('price')}"
            )

            lines.append(
                f"- 当日涨跌幅：{item.get('change_pct')}%"
            )

            lines.append(
                f"- 成交额：{item.get('amount')}"
            )

            lines.append(
                f"- RSI：{indicators.get('rsi')}"
            )

            lines.append(
                f"- MACD：{indicators.get('macd')}"
            )

            lines.append(
                f"- 量比：{indicators.get('volume_ratio')}"
            )

            lines.append(
                f"- 20日位置：{indicators.get('position_20')}%"
            )

            lines.append(
                f"- 60日位置：{indicators.get('position_60')}%"
            )

            lines.append(
                f"- 止盈参考：{item.get('参考止盈幅度')}%"
            )

            lines.append(
                f"- 止损参考：{item.get('参考止损幅度')}%"
            )

            risk_flags = item.get("risk_flags", [])

            lines.append(
                "- 风险提示：" +
                ("；".join(risk_flags) if risk_flags else "未发现明显技术风险提示")
            )

            lines.append(
                f"- 评分说明：{item.get('score_note')}"
            )

            lines.append("")

    lines.append("## 三、数据源状态")
    lines.append("")

    for key, value in source_status.items():
        lines.append(f"- **{key}**：{value}")

    lines.append("")

    lines.append("## 四、重要说明")
    lines.append("")

    lines.append(
        "本模型输出为量化筛选结果，不构成投资建议。"
        "评分不代表未来收益概率，也不保证次日上涨。"
    )

    lines.append(
        "当前数据源未提供可靠的全行业主力资金流向、委比和内外盘数据，"
        "相关字段不会用虚构数字填充。"
    )

    lines.append(
        "止盈止损仅为参考规则，实际成交可能受到跳空、流动性、"
        "交易限制和滑点影响。"
    )

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ============================================================
# 八、学习记录接口
# ============================================================

def save_learning_records(
    output_dir,
    run_id,
    recommendations,
    market_summary
):
    """
    保存本轮推荐，供 learning.py 后续复盘。

    learning.py 将在下一份文件中实现 record_scan。
    """

    try:
        from learning import record_scan

        record_scan(
            output_dir=output_dir,
            run_id=run_id,
            recommendations=recommendations,
            market_summary=market_summary
        )

        return True

    except Exception as exc:
        logging.getLogger("ashare_hunter").warning(
            "学习记录保存失败：%s",
            exc
        )

        return False


# ============================================================
# 九、主扫描流程
# ============================================================

def run_hunter(
    config_path="config.json",
    output_dir=None
):
    config = load_config(config_path)

    if output_dir is None:
        output_dir = config.get(
            "output", {}
        ).get("directory", "reports")

    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)

    run_id = make_run_id()

    started_at = now_text()

    reset_data_source_status()

    status = {
        "state": "运行中",
        "version": config.get("version"),
        "run_id": run_id,
        "started_at": started_at,
        "errors": []
    }

    save_json(
        os.path.join(output_dir, "status.json"),
        status
    )

    logger.info("=" * 60)
    logger.info("A股猎手开始运行")
    logger.info("扫描编号：%s", run_id)
    logger.info("=" * 60)

    try:
        # 1. 获取全市场行情。
        logger.info("步骤1：获取沪深A股全市场行情")

        snapshot = get_market_snapshot(config)

        if not snapshot:
            raise RuntimeError(
                "全市场行情为空，本轮停止"
            )

        logger.info(
            "全市场有效行情数量：%s",
            len(snapshot)
        )

        # 2. 市场环境分析。
        logger.info("步骤2：计算市场整体涨跌情况")

        market_summary = calculate_market_summary(
            snapshot
        )

        logger.info(
            "市场环境：%s，上涨比例：%s%%",
            market_summary["市场环境"],
            market_summary["上涨比例"]
        )

        # 3. 初步筛选。
        logger.info("步骤3：初步筛选候选股票")

        candidates = preliminary_screen(
            snapshot,
            config
        )

        logger.info(
            "初步筛选得到 %s 只候选股票",
            len(candidates)
        )

        if not candidates:
            raise RuntimeError(
                "初步筛选后没有候选股票。"
                "本轮不生成虚假的前三名。"
            )

        # 4. 获取历史K线并计算指标。
        logger.info("步骤4：获取候选股历史K线并评分")

        analyzed = analyze_candidates(
            candidates,
            config,
            market_summary,
            logger
        )

        if not analyzed:
            raise RuntimeError(
                "候选股票全部分析失败，本轮不输出虚假结果"
            )

        # 5. 选出前三名。
        minimum_score = float(
            config.get("screening", {}).get(
                "min_score",
                6.0
            )
        )

        top_n = int(
            config.get("screening", {}).get(
                "top_n",
                3
            )
        )

        qualified = [
            item for item in analyzed
            if item.get("score", 0) >= minimum_score
        ]

        recommendations = qualified[:top_n]

        # 6. 保存报告。
        logger.info("步骤5：保存扫描报告")

        save_candidates_csv(
            os.path.join(
                output_dir,
                f"candidates_{run_id}.csv"
            ),
            analyzed
        )

        save_json(
            os.path.join(
                output_dir,
                f"recommendations_{run_id}.json"
            ),
            recommendations
        )

        save_json(
            os.path.join(
                output_dir,
                f"market_{run_id}.json"
            ),
            market_summary
        )

        source_status = get_data_source_status()

        save_json(
            os.path.join(
                output_dir,
                f"data_source_{run_id}.json"
            ),
            source_status
        )

        save_markdown_report(
            os.path.join(
                output_dir,
                f"report_{run_id}.md"
            ),
            run_id,
            market_summary,
            analyzed,
            recommendations,
            source_status
        )

        # 7. 保存长期学习记录。
        learning_saved = save_learning_records(
            output_dir,
            run_id,
            recommendations,
            market_summary
        )

        finished_at = now_text()

        status = {
            "state": "运行成功",
            "version": config.get("version"),
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "market_count": len(snapshot),
            "candidate_count": len(candidates),
            "analyzed_count": len(analyzed),
            "qualified_count": len(qualified),
            "recommendation_count": len(recommendations),
            "learning_saved": learning_saved,
            "market_environment": market_summary["市场环境"],
            "data_source": source_status,
            "recommendations": [
                {
                    "code": item.get("code"),
                    "name": item.get("name"),
                    "score": item.get("score")
                }
                for item in recommendations
            ],
            "errors": []
        }

        save_json(
            os.path.join(output_dir, "status.json"),
            status
        )

        logger.info("=" * 60)
        logger.info("扫描完成")
        logger.info("成功分析：%s 只", len(analyzed))
        logger.info("达到最低分数：%s 只", len(qualified))

        if recommendations:
            for index, item in enumerate(recommendations, 1):
                logger.info(
                    "第%s名：%s %s，评分：%s",
                    index,
                    item.get("code"),
                    item.get("name"),
                    item.get("score")
                )
        else:
            logger.info(
                "本轮没有股票达到最低评分 %.2f",
                minimum_score
            )

        logger.info("=" * 60)

        return status

    except Exception as exc:

        logger.exception("扫描失败")

        status = {
            "state": "运行失败，未生成有效选股结果",
            "version": config.get("version"),
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": now_text(),
            "errors": [
                {
                    "error": f"{type(exc).__name__}: {exc}"
                }
            ],
            "data_source": get_data_source_status()
        }

        save_json(
            os.path.join(output_dir, "status.json"),
            status
        )

        raise


# ============================================================
# 十、命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="A股猎手全市场扫描程序"
    )

    parser.add_argument(
        "--config",
        default="config.json",
        help="配置文件路径"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="报告输出目录"
    )

    args = parser.parse_args()

    run_hunter(
        config_path=args.config,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
