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

def calculate_indicators(df):
    df = df.copy()
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
    raw_code = str(row["代码"])
    code = re.sub(r'\D', '', raw_code).zfill(6)
    name = str(row["名称"])
    
    try:
        hist = None
        try:
            hist = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=(NOW - timedelta(days=100)).strftime("%Y%m%d"),
                end_date=today_compact,
                adjust="qfq",
            )
        except Exception as e:
            # 备用新浪接口
            if code.startswith('6'):
                symbol_sina = f"sh{code}"
            elif code.startswith('0') or code.startswith('3'):
                symbol_sina = f"sz{code}"
            else:
                return None, f"{code} 代码格式不支持"
            
            hist = ak.stock_zh_a_daily(symbol=symbol_sina, adjust="qfq")
            hist = hist.rename(columns={'date': '日期', 'close': '收盘', 'volume': '成交量'})

        if hist is None or len(hist) < 30:
            return None, f"{code} 数据不足30天"
            
        last_date_str = str(hist.iloc[-1]["日期"]).replace("-", "")[:8]
        if last_date_str != today_compact:
            return None, f"{code} 数据未更新（最后一天:{last_date_str}）"

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

        score = 0.0
        reasons = []
        if price >= ma5: score += 1; reasons.append("收盘站上MA5")
        if ma5 >= ma10: score += 1; reasons.append("短期均线偏强")
        if price >= ma20: score += 1; reasons.append("收盘站上MA20")
        if macd > prev_macd: score += 1; reasons.append("MACD改善")
        if macd > 0: score += 0.5; reasons.append("MACD柱为正")
        if vol5 > 0 and vol < vol5 * 1.5: score += 0.5; reasons.append("成交量未明显失控")
        if 0 <= pct <= 3: score += 1; reasons.append("当日涨幅适中")
        if 1 <= turnover <= 10: score += 0.5; reasons.append("换手率处于观察区间")
        if pct > 5: score -= 2; reasons.append("当日涨幅较大，谨慎追高")

        candidate = {
            "代码": code, "名称": name, "收盘价": round(price, 3),
            "当日涨跌幅%": round(pct, 2), "成交额(元)": int(amount),
            "换手率%": round(turnover, 2), "MACD柱": round(macd, 4),
            "MA5": round(ma5, 3), "MA10": round(ma10, 3), "MA20": round(ma20, 3),
            "评分": round(score, 2), "理由": "；".join(reasons),
        }
        return candidate, None

    except Exception as exc:
        return None, f"{code} 异常: {str(exc)[:30]}"

def scan_market():
    log("开始检查今日是否为 A 股交易日……")
    try:
        trade_dates = ak.tool_trade_date_hist_sina()
        trade_dates["trade_date"] = pd.to_datetime(trade_dates["trade_date"]).dt.strftime("%Y-%m-%d")
        if TODAY not in trade_dates["trade_date"].values:
            log(f"今天 {TODAY} 不是A股交易日，退出。")
            return
    except Exception as e:
        log(f"交易日历获取失败: {e}")

    log("开始获取全市场行情...")
    spot = None
    try:
        spot = ak.stock_zh_a_spot_em()
    except Exception as e:
        log("东财接口失败，切换新浪...")
        spot = ak.stock_zh_a_spot()
        spot = spot.rename(columns={'代码': '代码', '名称': '名称', '最新价': '最新价', '涨跌幅': '涨跌幅', '成交额': '成交额'})
        if '换手率' not in spot.columns:
            spot['换手率'] = 1.0 

    for col in ["最新价", "涨跌幅", "成交额", "换手率"]:
        spot[col] = pd.to_numeric(spot[col], errors="coerce")
    spot = spot.dropna(subset=["最新价", "涨跌幅", "成交额", "换手率"])
    
    # ================= 核心改动：剔除北交所和科创板 =================
    log("开始剔除北交所(8,4,9开头)和科创板(688开头)股票...")
    # 提取纯数字代码用于判断
    spot["纯代码"] = spot["代码"].astype(str).str.extract(r'(\d+)')[0].fillna('')
    
    # 过滤条件：
    # 1. 不是ST或退市
    # 2. 不是北交所 (8, 4, 9开头)
    # 3. 不是科创板 (688开头)
    spot = spot[
        (~spot["名称"].str.contains("ST|退", na=False))
        & (~spot["纯代码"].str.startswith(('8', '4', '9')))
        & (~spot["纯代码"].str.startswith('688'))
        & (spot["最新价"] > 0)
        & (spot["成交额"] >= 3e7)  # 成交额大于3000万
        & (spot["换手率"] >= 0.5)  # 换手率大于0.5%
        & (spot["涨跌幅"] > -7)
        & (spot["涨跌幅"] < 7)
    ].copy()
    # =================================================================

    log(f"基础过滤(已剔除北交所/科创板)后剩余 {len(spot)} 只，开始计算……")

    candidates = []
    failures = []
    
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
                log(f"已处理 {completed_count}/{len(spot)}，候选 {len(candidates)}，失败 {len(failures)}")
                if len(failures) > 0:
                    log(f"   └─ 最近失败原因: {failures[-1]}")

    if not candidates:
        raise RuntimeError("没有取得有效候选数据，不能生成结论。")

    result = pd.DataFrame(candidates).sort_values(["评分", "成交额(元)"], ascending=[False, False])
    os.makedirs("reports", exist_ok=True)
    result.head(30).to_csv(f"reports/{TODAY}.csv", index=False, encoding="utf-8-sig")
    
    report = [f"# A股猎手收盘报告：{TODAY}\n", f"- 初筛股票：{len(spot)}", f"- 成功：{len(candidates)}", f"- 失败：{len(failures)}\n", "## 前十候选：\n"]
    for i, item in enumerate(result.head(10).to_dict("records"), start=1):
        report.append(f"### {i}. {item['名称']}（{item['代码']}）\n- 收盘价：{item['收盘价']}\n- 评分：{item['评分']}\n- 理由：{item['理由']}\n")
    
    with open("reports/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log("运行完成。")

if __name__ == "__main__":
    try:
        scan_market()
    except Exception:
        traceback.print_exc()
        raise
