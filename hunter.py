# -*- coding: utf-8 -*-
import os, sys, json, csv, time, argparse, logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from data_source import get_market_snapshot, get_history
from indicators import calculate_indicators

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("ashare_hunter")
    logger.setLevel(logging.INFO)
    if logger.handlers: return logger
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file: return json.load(file)

def safe_float(value, default=None):
    try:
        result = float(value)
        return result if not pd.isna(result) else default
    except: return default

def calculate_market_summary(snapshot):
    if not snapshot: raise RuntimeError("市场快照为空")
    up = sum(1 for item in snapshot if safe_float(item.get("change_pct"), 0) > 0)
    breadth = up / len(snapshot) if snapshot else 0
    env = "市场普涨" if breadth >= 0.65 else "市场偏强" if breadth >= 0.5 else "市场分化" if breadth >= 0.35 else "市场偏弱"
    return {"total_stocks": len(snapshot), "上涨比例": round(breadth * 100, 2), "市场环境": env}

def preliminary_screen(snapshot, config):
    candidates = []
    for item in snapshot:
        code = str(item.get("code", ""))
        price = safe_float(item.get("price"))
        pct = safe_float(item.get("change_pct"), 0.0)
        
        if price is None or price <= 0: 
            continue
        
        # 核心过滤：严格剔除当日涨停和跌停
        if code.startswith(("300", "301")):
            # 创业板涨跌幅限制为 20%
            if pct <= -19.5 or pct >= 19.5: 
                continue
        else:
            # 主板涨跌幅限制为 10%
            if pct <= -9.8 or pct >= 9.8: 
                continue

        candidates.append(item)
    
    candidates.sort(key=lambda x: safe_float(x.get("amount"), 0) or 0, reverse=True)
    shortlist_size = config.get("screening", {}).get("shortlist_size", 6000)
    return candidates[:shortlist_size]

def calculate_stock_score(stock, indicators, market_summary, config):
    score = 0.0
    if indicators.get("above_ma20"): score += 2.0
    if indicators.get("macd_positive"): score += 2.0
    if indicators.get("volume_ratio") and 1.0 <= indicators["volume_ratio"] <= 2.5: score += 2.0
    if 40 <= indicators.get("rsi", 0) <= 65: score += 2.0
    if -1 <= stock.get("change_pct", 0) <= 3: score += 2.0
    return {"score": round(min(10, score), 2)}

def analyze_one_stock(stock, config, market_summary):
    try:
        history = get_history(stock["code"], config)
        indicators = calculate_indicators(history, config)
        score_result = calculate_stock_score(stock, indicators, market_summary, config)
        result = dict(stock)
        result["indicators"] = indicators
        result.update(score_result)
        return result
    except: return None

def run_hunter(config_path, output_dir):
    config = load_config(config_path)
    logger = setup_logger(output_dir)
    
    logger.info("步骤1：获取全市场行情")
    snapshot = get_market_snapshot(config)
    logger.info(f"有效行情数量：{len(snapshot)}")

    market_summary = calculate_market_summary(snapshot)
    logger.info(f"市场环境：{market_summary['市场环境']}，上涨比例：{market_summary['上涨比例']}%")

    logger.info("步骤2：初步筛选（严格剔除当日涨停/跌停）")
    candidates = preliminary_screen(snapshot, config)
    logger.info(f"初筛得到 {len(candidates)} 只候选股（将全部进行深度分析）")
    if not candidates: raise RuntimeError("初筛后无候选股票")

    logger.info(f"步骤3：计算全市场技术指标与评分（并发20，预计耗时15分钟）")
    analyzed = []
    completed = 0
    total = len(candidates)
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_one_stock, stock, config, market_summary): stock for stock in candidates}
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res: analyzed.append(res)
            if completed % 200 == 0:
                logger.info(f"分析进度：{completed}/{total} ...")

    analyzed.sort(key=lambda x: safe_float(x.get("score"), 0), reverse=True)
    recommendations = analyzed[:3]

    logger.info("步骤4：保存报告")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(os.path.join(output_dir, f"candidates_{run_id}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "name", "price", "change_pct", "score"])
        writer.writeheader()
        for item in analyzed:
            writer.writerow({k: item.get(k) for k in writer.fieldnames})

    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# A股猎手报告 {run_id}\n\n市场环境：{market_summary['市场环境']}\n\n")
        for i, item in enumerate(recommendations, 1):
            f.write(f"{i}. {item['name']}({item['code']}) | 评分:{item['score']} | 价格:{item['price']}\n")

    logger.info("运行成功！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    run_hunter(args.config, args.output)
