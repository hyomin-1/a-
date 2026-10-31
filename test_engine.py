# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd

from indicators import calculate_indicators
from hunter import calculate_market_summary, preliminary_screen, calculate_stock_score

class TestIndicators(unittest.TestCase):
    def test_indicators(self):
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=100)
        df = pd.DataFrame({
            "date": dates, "open": 10.0, "close": 10.0, "high": 11.0, "low": 9.0, "volume": 1000.0
        })
        res = calculate_indicators(df, {})
        self.assertIsNotNone(res["close"])
        self.assertIsNotNone(res["ma20"])

class TestMarket(unittest.TestCase):
    def test_summary(self):
        snapshot = [{"change_pct": 1.0}, {"change_pct": -1.0}]
        res = calculate_market_summary(snapshot)
        self.assertEqual(res["total_stocks"], 2)
        self.assertEqual(res["上涨比例"], 50.0)

    def test_preliminary_screen(self):
        snapshot = [
            {"code": "600000", "price": 10.0, "change_pct": 1.0, "amount": 100.0},
            {"code": "600001", "price": 10.0, "change_pct": 10.0, "amount": 200.0}, # 主板涨停，剔除
            {"code": "300001", "price": 10.0, "change_pct": 15.0, "amount": 200.0}, # 创业板未涨停，保留
            {"code": "300002", "price": 10.0, "change_pct": 20.0, "amount": 200.0}  # 创业板涨停，剔除
        ]
        config = {"screening": {"shortlist_size": 6000}}
        res = preliminary_screen(snapshot, config)
        self.assertEqual(len(res), 2)
        codes = [x["code"] for x in res]
        self.assertIn("600000", codes)
        self.assertIn("300001", codes)
        self.assertNotIn("600001", codes)
        self.assertNotIn("300002", codes)

class TestScoring(unittest.TestCase):
    def test_score(self):
        stock = {"change_pct": 1.0}
        indicators = {"above_ma20": True, "macd_positive": True, "volume_ratio": 1.5, "rsi": 50}
        res = calculate_stock_score(stock, indicators, {}, {})
        self.assertEqual(res["score"], 10.0)

if __name__ == "__main__":
    unittest.main()
