import os
import time
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import numpy as np
import pandas as pd

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime.now(TZ)
TODAY = NOW.strftime("%Y-%m-%d")
TODAY_COMPACT = NOW.strftime("%Y%m%d")

def log(message):
    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] {message}", flush=True)

def calculate_indicators(df):
    df = df.copy()
    # 兼容列名
    close_col = "收盘" if "收盘" in df.columns else "close"
    vol_col = "成交量" if "成交量" in df.columns else "volume"
    
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
    df = df.dropna(subset=[close_col]).reset_index(drop=True)

    if len(df) < 30:
        return None
    
    close = df[close_col]
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    df["成交量均值5"] = df[vol_col].rolling(5).mean()
    df["成交量均值20"] = df[vol_col].rolling(20).mean()
    return df

def process_single_stock(row, today_compact):
    code = str(row["代码"]).zfill(6)
    name = str(row["名称"])
    
    try:
        # 获取近100天历史行情
        hist = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=(NOW - timedelta(days=100)).strftime("%Y%m%d"),
            end_date=today_compact,
            adjust="qfq",
        )
        if hist is None or len(hist) < 30:
            return None, f"{code} 数据不足30天"
            
        # 数据新鲜度校验：判断最后一天的数据是不是今天
        last_data_date = str(hist.iloc[-1]["日期"]).replace("-", "")
        if last_data_date != today_compact:
            return None, f"{code} 数据未更新（最后一天:{last_data_date}）"

        hist = calculate_indicators(hist)
        if hist is None:
            return None, f"{code} 指标计算失败"
            
        last = hist.iloc[-1]
        prev = hist.iloc[-2]
        
        price = float(last["收盘"])
        pct = float(row["涨跌幅"])
        turnover = float(row["换手率"])
        amount = float(row["成交额"])
        ma5 = float(last["MA5"])
        ma10 = float(last["MA10"])
        ma20 = float(last["MA20"])
        macd = float(last["MACD"])
        prev_macd = float(prev["MACD"])
        vol = float(last["成交量"])
        vol5 = float(last["成交量均值5"])
        
        if not all(np.isfinite(x) for x in [price, pct, turnover, amount, ma5, ma10, ma20, macd, prev_macd]):
            return None, f"{code} 指标包含无效值"

        # 隔夜策略评分逻辑
        score = 0.0
        reasons = []
        
        if price >= ma5:
            score += 1
            reasons.append("收盘站上MA5")
        if ma5 >= ma10:
            score += 1
            reasons.append("短期均线偏强")
        if price >= ma20:
            score += 1
            reasons.append("收盘站上MA20")
        if macd > prev_macd:
            score += 1
            reasons.append("MACD改善")
        if macd > 0:
            score += 0.5
            reasons.append("MACD柱为正")
        if vol5 > 0 and vol < vol5 * 1.5:
            score += 0.5
            reasons.append("成交量未明显失控")
        if 0 <= pct <= 3:
            score += 1
            reasons.append("当日涨幅适中")
        if 1 <= turnover <= 10:
            score += 0.5
            reasons.append("换手率处于观察区间")
        if pct > 5:
            score -= 2
            reasons.append("当日涨幅较大，谨慎追高")

        candidate = {
            "代码": code,
            "名称": name,
            "收盘价": round(price, 3),
            "当日涨跌幅%": round(pct, 2),
            "成交额(元)": int(amount),
            "换手率%": round(turnover, 2),
            "MACD柱": round(macd, 4),
            "MA5": round(ma5, 3),
            "MA10": round(ma10, 3),
            "MA20": round(ma20, 3),
            "评分": round(score, 2),
            "理由": "；".join(reasons),
        }
        return candidate, None

    except Exception as exc:
        return None, f"{code} 异常: {exc}"

def scan_market():
    log("开始检查今日是否为 A 股交易日……")
    try:
        # 获取交易日历校验
        trade_dates = ak.tool_trade_date_hist_sina()
        trade_dates["trade_date"] = pd.to_datetime(trade_dates["trade_date"]).dt.strftime("%Y-%m-%d")
        if TODAY not in trade_dates["trade_date"].values:
            log(f"今天 {TODAY} 不是A股交易日（可能是周末或节假日），程序安全退出。")
            return
    except Exception as e:
        log(f"获取交易日历失败，跳过校验继续执行：{e}")

    log("获取沪深京A股全市场行情……")
    spot = ak.stock_zh_a_spot_em()

    # 安全过滤机制，使用列名索引
    required = {"代码", "名称", "最新价", "涨跌幅", "成交额", "换手率"}
    if not required.issubset(set(spot.columns)):
        raise RuntimeError(f"行情字段缺失，当前接口返回：{spot.columns.tolist()}")

    for col in ["最新价", "涨跌幅", "成交额", "换手率"]:
        spot[col] = pd.to_numeric(spot[col], errors="coerce")

    spot = spot.dropna(subset=["最新价", "涨跌幅", "成交额", "换手率"])
    spot = spot[
        (~spot["名称"].str.contains("ST|退", na=False))
        & (spot["最新价"] > 0)
        & (spot["成交额"] >= 3e7)  # 成交额大于3000万
        & (spot["换手率"] >= 0.5)  # 换手率大于0.5%
        & (spot["涨跌幅"] > -7)
        & (spot["涨跌幅"] < 7)
    ].copy()
    
    log(f"基础条件过滤后剩余 {len(spot)} 只，开始并发计算历史指标……")

    candidates = []
    failures = []
    
    # 引入多线程并发，限制 max_workers=5 防止被封 IP
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_stock, row, TODAY_COMPACT): row for _, row in spot.iterrows()}
        
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            candidate, error_msg = future.result()
            
            if candidate:
                candidates.append(candidate)
            elif error_msg:
                failures.append(error_msg)
                
            if completed_count % 50 == 0:
                log(f"已处理 {completed_count}/{len(spot)}，有效候选 {len(candidates)}，失败 {len(failures)}")

    if not candidates:
        raise RuntimeError("没有取得有效候选数据，不能生成选股结论。可能是数据源未更新。")

    # 排序并生成报告
    result = pd.DataFrame(candidates).sort_values(
        ["评分", "成交额(元)"], ascending=[False, False]
    )
    
    os.makedirs("reports", exist_ok=True)
    path = f"reports/{TODAY}.csv"
    result.head(30).to_csv(path, index=False, encoding="utf-8-sig")
    
    report = [
        f"# A股猎手收盘报告：{TODAY}",
        "",
        f"- 运行时间：{NOW.strftime('%Y-%m-%d %H:%M:%S')}（北京时间）",
        f"- 初筛股票：{len(spot)}",
        f"- 成功计算指标：{len(candidates)}",
        f"- 获取失败或数据不足：{len(failures)}",
        "",
        "## 排名前十的候选股",
        "",
    ]
    
    for i, item in enumerate(result.head(10).to_dict("records"), start=1):
        report.extend([
            f"### {i}. {item['名称']}（{item['代码']}）",
            f"- 收盘价：{item['收盘价']}",
            f"- 当日涨跌幅：{item['当日涨跌幅%']}%",
            f"- 换手率：{item['换手率%']}%",
            f"- MACD柱：{item['MACD柱']}",
            f"- 综合评分：{item['评分']}",
            f"- 参考理由：{item['理由']}",
            "",
        ])
        
    report.extend([
        "## 风险说明",
        "",
        "- 这是基于公开行情和技术指标的初筛，不代表预测必然有效。",
        "- 行情源可能延迟、缺失或连接失败；必须核对实际交易日期。",
        "- 当前评分没有经过历史回测，不应直接视为买入信号。",
        "- 次日开盘低开、突发公告、流动性和交易成本都可能造成亏损。",
        "- 若数据不完整，应停止输出有效选股结论。",
        "",
    ])
    
    with open("reports/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    log(f"报告已保存：{path}")
    log("运行完成。")

if __name__ == "__main__":
    try:
        scan_market()
    except Exception:
        log("运行失败，未能生成可信选股结果。")
        traceback.print_exc()
        raise
