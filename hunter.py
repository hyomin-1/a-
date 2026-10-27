import os
import re
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

# 计算 RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    df = df.copy()
    close_col = "收盘" if "收盘" in df.columns else "close"
    vol_col = "成交量" if "成交量" in df.columns else "volume"
    
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
    df = df.dropna(subset=[close_col]).reset_index(drop=True)

    if len(df) < 20: return None
    
    close = df[close_col]
    # 核心均线：10日线
    df["MA10"] = close.rolling(10).mean()
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    
    # RSI
    df["RSI"] = calculate_rsi(close)
    
    return df

def get_hot_sectors():
    """获取今日资金净流入前10的行业板块"""
    try:
        log("正在获取今日资金热门板块...")
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        top_sectors = df.head(10)["名称"].tolist()
        log(f"今日资金流入前十板块: {top_sectors}")
        return top_sectors
    except Exception as e:
        log(f"板块资金获取失败: {e}")
        return []

def process_single_stock(row, today_compact, hot_sectors):
    raw_code = str(row["代码"])
    code = re.sub(r'\D', '', raw_code).zfill(6)
    name = str(row["名称"])
    
    try:
        # 获取历史数据 (60天)
        hist = None
        try:
            hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=(NOW - timedelta(days=60)).strftime("%Y%m%d"), end_date=today_compact, adjust="qfq")
        except Exception:
            if code.startswith('6'): symbol_sina = f"sh{code}"
            elif code.startswith('0') or code.startswith('3'): symbol_sina = f"sz{code}"
            else: return None
            hist = ak.stock_zh_a_daily(symbol=symbol_sina, adjust="qfq")
            hist = hist.rename(columns={'date': '日期', 'close': '收盘', 'volume': '成交量'})

        if hist is None or len(hist) < 20: return None
            
        hist = calculate_indicators(hist)
        if hist is None: return None
            
        last = hist.iloc[-1]
        prev = hist.iloc[-2]
        
        # 历史指标
        ma10 = float(last["MA10"])
        macd = float(last["MACD"])
        prev_macd = float(prev["MACD"])
        rsi = float(last["RSI"])
        
        # 实时盘口指标
        price = float(row["最新价"])
        pct = float(row["涨跌幅"])
        volume_ratio = float(row["量比"]) if not pd.isna(row["量比"]) else 0
        weibi = float(row["委比"]) if not pd.isna(row["委比"]) else 0
        outer = float(row["外盘"]) if not pd.isna(row["外盘"]) else 0
        inner = float(row["内盘"]) if not pd.isna(row["内盘"]) else 0
        
        # ================= 10分制评分系统 =================
        score = 0.0
        
        # 1. 趋势 (2分)：站上10日线
        if price > ma10: 
            score += 2
            
        # 2. MACD (2分)
        if macd > prev_macd: score += 1
        if macd > 0: score += 1
            
        # 3. RSI 强势区 (1.5分)：50~75属于主升浪，超过80属于超买风险
        if 50 <= rsi <= 75: 
            score += 1.5
            
        # 4. 盘口量价 (2.5分)
        if 1.2 <= volume_ratio <= 3.0: score += 1  # 温和放量
        if weibi > 10: score += 0.5                # 买盘压单
        if outer > inner: score += 1               # 主动买大于主动卖
            
        # 5. 板块资金与形态 (2分)
        # 检查是否在热门板块内 (粗略匹配，通过东财个股行业属性)
        is_hot = False
        # 为了速度，这里用涨幅和量比简单替代，实际应用可引入东财板块API匹配
        if 1.0 <= pct <= 4.5:
            score += 2
            
        # 基础门槛：没站上10日线或量比不足1的直接淘汰
        if price <= ma10 or volume_ratio < 1.0:
            return None

        # ================= 止盈止损 =================
        # 目标：次日溢价 1.5% (稳清仓) - 2%
        target_price = round(price * 1.015, 2) 
        stop_loss = round(price * 0.98, 2)     # 严格 2% 止损

        candidate = {
            "代码": code, "名称": name, "最新价": price,
            "涨幅%": round(pct, 2), "量比": round(volume_ratio, 2),
            "委比": round(weibi, 1), "外/内": round(outer/inner, 2) if inner>0 else 99,
            "RSI": round(rsi, 1), "评分": round(score, 1),
            "次日止盈": target_price, "止损": stop_loss
        }
        return candidate

    except Exception:
        return None

def scan_market():
    # 1. 检查交易日
    try:
        trade_dates = ak.tool_trade_date_hist_sina()
        trade_dates["trade_date"] = pd.to_datetime(trade_dates["trade_date"]).dt.strftime("%Y-%m-%d")
        if TODAY not in trade_dates["trade_date"].values:
            log(f"今天 {TODAY} 不是A股交易日，退出。")
            return
    except Exception: pass

    # 2. 获取热门板块资金
    hot_sectors = get_hot_sectors()

    # 3. 获取全市场快照
    log("获取全市场实时盘口数据...")
    try:
        spot = ak.stock_zh_a_spot_em()
    except Exception:
        log("东财快照失败，切新浪兜底...")
        spot = ak.stock_zh_a_spot().rename(columns={'代码': '代码', '名称': '名称', '最新价': '最新价', '涨跌幅': '涨跌幅', '成交额': '成交额', '成交量': '成交量'})
        for col in ["换手率", "量比", "委比", "外盘", "内盘"]:
            if col not in spot.columns: spot[col] = np.nan

    # 4. 基础剔除（去掉ST、北交所、科创板、成交额太小、绿盘的）
    spot = spot.dropna(subset=["最新价", "涨跌幅", "成交额"])
    spot["纯代码"] = spot["代码"].astype(str).str.extract(r'(\d+)')[0].fillna('')
    
    spot = spot[
        (~spot["名称"].str.contains("ST|退", na=False))
        & (~spot["纯代码"].str.startswith(('8', '4', '9', '688')))
        & (spot["最新价"] > 2) 
        & (spot["成交额"] >= 50000000)  # 成交额大于5000万，保证流动性
        & (spot["涨跌幅"] > 0) & (spot["涨跌幅"] < 6.0) # 尾盘只做红盘
    ].copy()

    log(f"基础过滤后剩余 {len(spot)} 只，开始深度计算...")

    # 5. 并发精算
    candidates = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_stock, row, TODAY_COMPACT, hot_sectors): row for _, row in spot.iterrows()}
        for future in as_completed(futures):
            res = future.result()
            if res: candidates.append(res)

    if not candidates:
        raise RuntimeError("今日无符合条件的尾盘狙击标的。")

    # 6. 排序输出
    df_all = pd.DataFrame(candidates)
    # 必须 >= 7 分才入选，宁缺毋滥
    df_top = df_all[df_all["评分"] >= 7.0].sort_values(["评分", "量比"], ascending=[False, False])
    result = df_top.head(3) # 只选前3支
    
    os.makedirs("reports", exist_ok=True)
    if len(result) == 0:
        log("⚠️ 今日无股票达到 7.0 分及格线，建议空仓休息。")
        return
        
    result.to_csv(f"reports/{TODAY}.csv", index=False, encoding="utf-8-sig")
    
    report = [f"# 尾盘狙击计划：{TODAY}\n", f"- 达标标的：{len(result)} 只\n"]
    for i, item in enumerate(result.to_dict("records"), start=1):
        report.append(f"### {i}. {item['名称']}（{item['代码']}） | 评分: {item['评分']}")
        report.append(f"- **当前价**：{item['最新价']} (涨幅 {item['涨幅%']}%)")
        report.append(f"- 🎯 **次日止盈**：**{item['次日止盈']}** (+1.5% 离场)")
        report.append(f"- 🛑 **严格止损**：**{item['止损']}** (-2.0% 破位走人)")
        report.append(f"- 📊 盘口数据：量比 {item['量比']} | 委比 {item['委比']}% | 外/内 {item['外/内']} | RSI {item['RSI']}\n")
        
    with open("reports/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log("运行完成，请查看报告。")

if __name__ == "__main__":
    try:
        scan_market()
    except Exception:
        traceback.print_exc()
