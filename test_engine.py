
# -*- coding: utf-8 -*-
"""
A股猎手 - 离线自动测试

不连接新浪、腾讯，不访问外部网络。
使用模拟行情和模拟K线检查项目核心功能。
"""

import unittest
import tempfile

import numpy as np
import pandas as pd

from indicators import calculate_indicators

from hunter import (
    calculate_market_summary,
    preliminary_screen,
    calculate_stock_score
)

from learning import (
    classify_stock_pattern,
    load_learning_memory,
    record_scan,
    get_learning_adjustment
)


TEST_CONFIG = {
    "screening": {
        "shortlist_size": 60,
        "top_n": 3,
        "min_score": 6.0,
        "min_price": 3.0,
        "max_price": 150.0,
        "min_amount_yuan": 100000000,
        "min_day_pct": -1.0,
        "max_day_pct": 5.5
    },
    "indicators": {
        "ma_periods": [5, 10, 20, 60],
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_period": 14
    },
    "scoring": {
        "maximum_score": 10.0,
        "minimum_score": 6.0,
        "technical_weight": 2.5,
        "price_volume_weight": 1.5,
        "low_position_weight": 1.5,
        "market_environment_weight": 1.0,
        "max_learning_penalty": 1.0
    },
    "learning": {
        "enabled": True,
        "apply_learning": True,
        "learning_window_sessions": 60,
        "min_learning_samples": 30,
        "min_learning_dates": 10,
        "min_learning_group": 12,
        "max_learning_penalty": 1.0,
        "review_after_sessions": [1, 5, 10]
    },
    "risk": {
        "reference_take_profit_pct": 3.0,
        "reference_stop_loss_pct": 2.0,
        "max_daily_rise_pct": 5.5,
        "avoid_high_rsi": 75,
        "avoid_extreme_volume_ratio": 4.0
    },
    "data_source": {
        "max_workers": 4
    },
    "output": {
        "directory": "reports"
    }
}


def make_history(rows=220):
    """生成模拟日K线。"""

    np.random.seed(42)

    dates = pd.bdate_range(
        end=pd.Timestamp.today(),
        periods=rows
    )

    close = []
    price = 15.0

    for index in range(rows):
        daily_change = (
            0.0008
            + np.random.normal(0, 0.012)
        )

        price *= (1 + daily_change)
        close.append(price)

    close = np.array(close)

    open_price = close * (
        1 + np.random.normal(0, 0.004, rows)
    )

    high = np.maximum(
        open_price,
        close
    ) * (
        1 + np.random.uniform(0.001, 0.02, rows)
    )

    low = np.minimum(
        open_price,
        close
    ) * (
        1 - np.random.uniform(0.001, 0.02, rows)
    )

    volume = np.random.randint(
        1000000,
        5000000,
        rows
    ).astype(float)

    return pd.DataFrame({
        "date": dates,
        "open": open_price,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "amount": volume * close
    })


class TestIndicators(unittest.TestCase):

    def setUp(self):
        self.history = make_history()
        self.config = TEST_CONFIG

    def test_indicators_not_empty(self):
        result = calculate_indicators(
            self.history,
            self.config
        )

        self.assertIsInstance(result, dict)
        self.assertTrue(result)

    def test_moving_averages(self):
        result = calculate_indicators(
            self.history,
            self.config
        )

        self.assertIsNotNone(result["ma5"])
        self.assertIsNotNone(result["ma10"])
        self.assertIsNotNone(result["ma20"])
        self.assertIsNotNone(result["ma60"])

    def test_macd_exists(self):
        result = calculate_indicators(
            self.history,
            self.config
        )

        self.assertIsNotNone(result["dif"])
        self.assertIsNotNone(result["dea"])
        self.assertIsNotNone(result["macd"])

    def test_rsi_range(self):
        result = calculate_indicators(
            self.history,
            self.config
        )

        self.assertIsNotNone(result["rsi"])

        self.assertGreaterEqual(
            result["rsi"],
            0
        )

        self.assertLessEqual(
            result["rsi"],
            100
        )

    def test_position_range(self):
        result = calculate_indicators(
            self.history,
            self.config
        )

        if result["position_20"] is not None:
            self.assertGreaterEqual(
                result["position_20"],
                0
            )

            self.assertLessEqual(
                result["position_20"],
                100
            )


class TestMarketAnalysis(unittest.TestCase):

    def setUp(self):

        self.snapshot = [
            {
                "code": "600000",
                "name": "测试银行",
                "price": 10,
                "prev_close": 9.8,
                "change_pct": 2.04,
                "amount": 200000000
            },
            {
                "code": "000001",
                "name": "测试股票",
                "price": 8,
                "prev_close": 8.2,
                "change_pct": -2.44,
                "amount": 300000000
            },
            {
                "code": "600001",
                "name": "测试股份",
                "price": 12,
                "prev_close": 12,
                "change_pct": 0,
                "amount": 150000000
            }
        ]

    def test_market_summary(self):

        result = calculate_market_summary(
            self.snapshot
        )

        self.assertEqual(
            result["total_stocks"],
            3
        )

        self.assertEqual(
            result["上涨家数"],
            1
        )

        self.assertEqual(
            result["下跌家数"],
            1
        )

        self.assertEqual(
            result["平盘家数"],
            1
        )

        self.assertFalse(
            result["sector_fund_flow_available"]
        )

    def test_preliminary_screen(self):

        snapshot = [
            {
                "code": "600000",
                "name": "测试银行",
                "price": 10,
                "change_pct": 2,
                "amount": 200000000
            },
            {
                "code": "000001",
                "name": "价格过高",
                "price": 200,
                "change_pct": 2,
                "amount": 300000000
            },
            {
                "code": "600001",
                "name": "成交额不足",
                "price": 12,
                "change_pct": 1,
                "amount": 10000000
            }
        ]

        result = preliminary_screen(
            snapshot,
            TEST_CONFIG
        )

        self.assertEqual(
            len(result),
            1
        )

        self.assertEqual(
            result[0]["code"],
            "600000"
        )


class TestScoring(unittest.TestCase):

    def test_score_range(self):

        stock = {
            "code": "600000",
            "name": "测试股票",
            "price": 15,
            "change_pct": 1.5,
            "amount": 300000000
        }

        indicators = {
            "above_ma20": True,
            "above_ma60": True,
            "ma20_rising": True,
            "ma_bullish": True,
            "macd_positive": True,
            "macd_improving": True,
            "macd_golden_cross": False,
            "rsi": 55,
            "volume_ratio": 1.5,
            "position_20": 45,
            "position_60": 40,
            "risk_count": 0,
            "risk_flags": []
        }

        market = {
            "上涨比例": 60,
            "市场环境": "市场偏强"
        }

        result = calculate_stock_score(
            stock,
            indicators,
            market,
            TEST_CONFIG
        )

        self.assertGreaterEqual(
            result["score"],
            0
        )

        self.assertLessEqual(
            result["score"],
            10
        )

        self.assertIsNone(
            result["sector_score"]
        )

        self.assertIsNone(
            result["order_book_score"]
        )


class TestLearning(unittest.TestCase):

    def test_pattern_classification(self):

        stock = {
            "change_pct": 1.5
        }

        indicators = {
            "position_60": 25,
            "rsi": 48,
            "volume_ratio": 1.5,
            "above_ma20": True,
            "macd_positive": True
        }

        pattern = classify_stock_pattern(
            stock,
            indicators
        )

        self.assertIsInstance(
            pattern,
            str
        )

        self.assertIn(
            "60日低位",
            pattern
        )

    def test_learning_memory_initialization(self):

        # 默认记忆可以只在内存中初始化，
        # 不强制要求立即创建 JSON 文件。
        with tempfile.TemporaryDirectory() as temp_dir:

            memory = load_learning_memory(
                temp_dir
            )

            self.assertIsInstance(
                memory,
                dict
            )

            self.assertIn(
                "patterns",
                memory
            )

            self.assertIsInstance(
                memory["patterns"],
                dict
            )

    def test_record_scan(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            stock = {
                "code": "600000",
                "name": "测试股票",
                "price": 15,
                "change_pct": 1.5,
                "score": 7.5,
                "indicators": {
                    "rsi": 50,
                    "macd": 0.2,
                    "volume_ratio": 1.5,
                    "position_20": 30,
                    "position_60": 25,
                    "above_ma20": True,
                    "macd_positive": True
                }
            }

            records = record_scan(
                output_dir=temp_dir,
                run_id="TEST001",
                recommendations=[stock],
                market_summary={
                    "市场环境": "市场偏强"
                }
            )

            self.assertEqual(
                len(records),
                1
            )

            self.assertEqual(
                records[0]["code"],
                "600000"
            )

    def test_learning_adjustment_without_samples(self):

        memory = {
            "patterns": {}
        }

        adjustment = get_learning_adjustment(
            {
                "change_pct": 1
            },
            {
                "position_60": 30,
                "rsi": 50,
                "volume_ratio": 1.5,
                "above_ma20": True,
                "macd_positive": True
            },
            memory,
            TEST_CONFIG
        )

        self.assertEqual(
            adjustment,
            0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
