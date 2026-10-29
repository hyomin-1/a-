        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                res = future.result()
                if res and "_error" in res:
                    failed += 1
                    errors.append({"时间": sh_now().strftime("%Y-%m-%d %H:%M:%S"), "错误": res["_error"]})
                elif res:
                    success += 1
                    candidates.append(res)
                else:
                    # 被策略过滤（如ST）或低于最低评分，不算接口失败
                    pass
            except Exception as exc:
                failed += 1
                row = futures[future]
                errors.append({
                    "时间": sh_now().strftime("%Y-%m-%d %H:%M:%S"),
                    "错误": f"{row.get('代码', '')} {row.get('名称', '')}: {type(exc).__name__}: {exc}"
                })
            if completed % 200 == 0 or completed == total:
                log(f"进度：{completed}/{total}；候选 {len(candidates)}；失败 {failed}")

    # 保存错误记录，便于诊断接口限流/字段变化
    error_path = os.path.join(REPORT_DIR, "scan_errors.csv")
    if errors:
        err_df = pd.DataFrame(errors)
        write_header = not os.path.exists(error_path)
        err_df.to_csv(error_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")
    else:
        log("本轮没有记录到单股接口异常。")

    if board_df is not None and not board_df.empty:
        board_df.to_csv(os.path.join(REPORT_DIR, f"sector_flow_{today}.csv"), index=False, encoding="utf-8-sig")

    if review_df is not None and not review_df.empty:
        review_df.to_csv(os.path.join(REPORT_DIR, f"review_{today}.csv"), index=False, encoding="utf-8-sig")
    if learning_df is not None and not learning_df.empty:
        learning_df.to_csv(os.path.join(REPORT_DIR, f"learning_review_{today}.csv"), index=False, encoding="utf-8-sig")

    if candidates:
        candidates = apply_learned_penalties(candidates)
        df_all = pd.DataFrame(candidates)
        df_all = df_all.sort_values(["评分", "量比"], ascending=[False, False])
        result = df_all.head(TOP_N).copy()
        # 当日候选全量与最终前三分开保存
        df_all.to_csv(os.path.join(REPORT_DIR, f"candidates_{today}.csv"), index=False, encoding="utf-8-sig")
        result.to_csv(os.path.join(REPORT_DIR, f"{today}.csv"), index=False, encoding="utf-8-sig")

        # 累积推荐历史，按日期+代码去重，便于后续T+1/T+5复盘
        history_path = os.path.join(REPORT_DIR, "recommendation_history.csv")
        history = result.copy()
        history.insert(0, "推荐日期", today)
        if os.path.exists(history_path):
            try:
                old_hist = pd.read_csv(history_path, dtype={"代码": str})
                history = pd.concat([old_hist, history], ignore_index=True)
                history["代码"] = history["代码"].map(normalize_code)
                history = history.drop_duplicates(["推荐日期", "代码"], keep="last")
            except Exception as exc:
                log(f"读取推荐历史失败，将尝试重建：{exc}")
        history.to_csv(history_path, index=False, encoding="utf-8-sig")
    else:
        result = pd.DataFrame()
        # 关键：仍然输出空结果与状态报告，不能把接口故障误报为“无股票”
        pd.DataFrame(columns=["代码", "名称", "评分"]).to_csv(
            os.path.join(REPORT_DIR, f"{today}.csv"), index=False, encoding="utf-8-sig"
        )

    # 保存每次运行的独立快照：同一天多次运行也能复盘上一次结果
    if result is not None and not result.empty:
        run_dir = os.path.join(REPORT_DIR, "runs")
        os.makedirs(run_dir, exist_ok=True)
        run_stamp = sh_now().strftime("%Y-%m-%d_%H%M%S")
        result.to_csv(os.path.join(run_dir, f"{run_stamp}.csv"), index=False, encoding="utf-8-sig")

    stats = {
        "snapshot_source": snapshot_source,
        "universe": total,
        "success": success,
        "failed": failed,
        "seconds": round(time.time() - started, 1),
    }
    write_report(
        today=today,
        candidates_count=len(candidates),
        result=result,
        previous_date=previous_date,
        review_df=review_df,
        board_df=board_df,
        statuses=board_status,
        scan_stats=stats,
    )
    log(f"运行完成：候选 {len(candidates)}，最终推荐 {len(result)}，失败 {failed}，耗时 {stats['seconds']} 秒。")
    log(f"请查看 {REPORT_DIR}/report.md、{REPORT_DIR}/{today}.csv 和 {REPORT_DIR}/scan_errors.csv")


if __name__ == "__main__":
    try:
        scan_market()
    except Exception as exc:
        ensure_dirs()
        log(f"程序运行失败：{type(exc).__name__}: {exc}")
        traceback.print_exc()
        with open(os.path.join(REPORT_DIR, "fatal_error.log"), "a", encoding="utf-8") as f:
            f.write(f"\n[{sh_now().isoformat()}] {type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc())
