import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import engine as e


def minute(ts, close=10.1, low=None, high=None):
    return {"时间": ts, "收盘": close, "最高": high or close + .01,
            "最低": low or close - .01, "成交量": 1000,
            "成交额": close * 100000, "均价": 10.05}


def morning(day="2026-10-29", close=10.1):
    return [minute((e.at(day, "09:31:00") + timedelta(minutes=i)).isoformat(), close) for i in range(60)]


def signal():
    return {"signal_date": "2026-10-28", "code": "600000", "name": "模拟样本",
            "entry_proxy_price": 10, "tags": ["尾盘15分钟走弱"], "strategy_id": e.config_id(e.DEFAULTS)}


class EngineTests(unittest.TestCase):
    def test_tiny_gross_premium_is_not_net_profit(self):
        self.assertLess(e.net_pct(10, 10.01, e.DEFAULTS), 0)

    def test_first_positive_close_not_next_days_high(self):
        rows = morning(close=10.05)
        rows[0] = minute(rows[0]["时间"], 10.05, high=11)
        result = e.review_signal(signal(), rows, "2026-10-29", e.DEFAULTS)
        self.assertTrue(result["success"])
        self.assertTrue(result["exit_time"].startswith("2026-10-29T09:31"))
        self.assertEqual(result["exit_reference"], 10.05)
        self.assertNotEqual(result["exit_reference"], 11)

    def test_stop_precedes_a_later_recovery(self):
        rows = morning(close=10.10)
        rows[0] = minute(rows[0]["时间"], 9.80)
        result = e.review_signal(signal(), rows, "2026-10-29", e.DEFAULTS)
        self.assertFalse(result["success"])
        self.assertLess(result["net_return_pct"], -1.5)

    def test_time_exit_when_no_profit(self):
        result = e.review_signal(signal(), morning(close=10), "2026-10-29", e.DEFAULTS)
        self.assertFalse(result["success"])
        self.assertTrue(result["exit_time"].startswith("2026-10-29T10:30"))

    def test_missing_bars_do_not_become_failures(self):
        with self.assertRaisesRegex(ValueError, "不完整"):
            e.review_signal(signal(), morning()[1:], "2026-10-29", e.DEFAULTS)

    def test_flat_limit_like_bar_does_not_claim_fill(self):
        rows = morning(close=9.8)
        rows[0]["最高"] = rows[0]["最低"] = 9.8
        result = e.review_signal(signal(), rows, "2026-10-29", e.DEFAULTS)
        self.assertEqual(result["status"], "execution_unverified")
        self.assertIsNone(result["success"])

    def test_only_next_session_and_known_timestamps(self):
        with self.assertRaisesRegex(ValueError, "晚于"):
            e.review_signal(signal(), morning("2026-10-28"), "2026-10-28", e.DEFAULTS)
        bars = e.clean_bars(morning(), "2026-10-29", e.at("2026-10-29", "09:32:00"))
        self.assertEqual(len(bars), 2)

    def test_timezone_conversion(self):
        self.assertEqual(e.stamp("2026-10-29T06:30:00Z"), e.at("2026-10-29", "14:30:00"))

    def test_small_sample_does_not_change_rules(self):
        row = dict(signal(), review_session="2026-10-29", source="minute_proxy", status="proxy_complete", success=False, net_return_pct=-2)
        self.assertEqual(e.learn([row], e.DEFAULTS, "2026-10-29")["rules"], [])

    def test_learning_compares_both_groups_and_ignores_future(self):
        rows = []
        for i in range(12):
            day = "2026-10-" + str(i + 1).zfill(2)
            for j in range(3):
                rows.append(dict(signal(), signal_date=day, code=str(j), review_session="2026-10-20",
                                 source="minute_proxy", status="proxy_complete", success=j == 2,
                                 net_return_pct=-2 if j < 2 else 1,
                                 tags=["尾盘15分钟走弱"] if j < 2 else []))
        result = e.learn(rows, e.DEFAULTS, "2026-10-29")
        self.assertEqual(result["samples"], 36)
        self.assertGreater(result["rules"][0]["penalty"], 0)
        self.assertEqual(e.learn(rows, e.DEFAULTS, "2026-10-19")["samples"], 0)
        changed = dict(e.DEFAULTS, exit_time="10:00:00")
        self.assertEqual(e.learn(rows, changed, "2026-10-29")["samples"], 0)

    def test_no_learning_from_unverified_or_changed_rules(self):
        row = dict(signal(), review_session="2026-10-29", source="minute_proxy", status="execution_unverified", success=None)
        self.assertEqual(e.learn([row], e.DEFAULTS, "2026-10-29")["samples"], 0)

    def test_actual_minute_features_and_no_future_leak(self):
        day = "2026-10-29"
        rows = []
        for start, count in [("09:30:00", 121), ("13:00:00", 108)]:
            rows.extend(minute((e.at(day, start) + timedelta(minutes=i)).isoformat()) for i in range(count))
        rows.append(minute(day + " 15:00:00", 99))  # future bar must never be used
        history = [{"date": (e.at("2026-10-28", "00:00:00") - timedelta(days=i)).date().isoformat(), "close": 10} for i in range(45)]
        snapshot = {"代码": "600000", "名称": "模拟样本", "昨收": 10, "换手率": 4}
        cfg = dict(e.DEFAULTS, _previous_session="2026-10-28")
        result = e.analyse(snapshot, rows, history, e.at(day, "14:48:10"), cfg, .6)
        self.assertEqual(result["reference_price"], 10.1)
        self.assertLessEqual(result["score"], 10)
        with self.assertRaisesRegex(ValueError, "过期"):
            e.analyse(snapshot, rows, history, e.at(day, "14:55:00"), cfg, .6)

    def test_timeout_really_stops_hanging_process(self):
        start = time.monotonic()
        result, errors = e.bounded_jobs([{"id": "hung", "kind": "test_hang", "seconds": 30}],
                                      e.now() + timedelta(seconds=.5), dict(e.DEFAULTS, request_timeout_seconds=.2))
        self.assertEqual(result, {})
        self.assertTrue(errors)
        self.assertLess(time.monotonic() - start, 4)

    def test_daily_analyse_works_outside_fixed_1430_window(self):
        history=[]
        base=e.at("2026-09-01","00:00:00")
        for i in range(55):
            day=(base+timedelta(days=i)).date().isoformat()
            close=10+i*0.03
            history.append({"日期":day,"收盘":close,"开盘":close,"最高":close*1.01,
                            "最低":close*.99,"成交额":200000000,"涨跌幅":0.3,"换手率":3})
        snapshot={"代码":"600000","名称":"模拟样本","最新价":11.8,"昨收":11.7,
                  "成交额":300000000,"换手率":4}
        row=e.daily_analyse(snapshot,history,e.at("2026-10-30","20:00:00").isoformat(),e.DEFAULTS,.6)
        self.assertIsNotNone(row)
        self.assertEqual(row["data_mode"],"latest_completed_daily")
        self.assertTrue(row["signal_time"].endswith("T15:00:00+08:00"))


    def test_idempotent_previous_friday_review(self):
        cfg = copy.deepcopy(e.DEFAULTS)
        prev = dict(signal(), signal_date="2026-10-23")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            e.save_json(root / "signals" / "2026-10-23.json", [prev])
            with patch.object(e, "bounded_jobs", return_value=({"600000": morning("2026-10-26")}, [])):
                for _ in range(2):
                    e.review_previous(root, "2026-10-26", ["2026-10-23", "2026-10-26"], cfg, e.now())
            rows = e.load_json(root / "learning_outcomes.json", [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["review_session"], "2026-10-26")


if __name__ == "__main__":
    unittest.main()
