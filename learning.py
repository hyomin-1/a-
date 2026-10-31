# -*- coding: utf-8 -*-
import os, json
def load_learning_memory(output_dir="reports"):
    path = os.path.join(output_dir, "learning_memory.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
        except: pass
    return {"patterns": {}, "statistics": {}}

def record_scan(output_dir, run_id, recommendations, market_summary):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "learning_memory.json")
    memory = load_learning_memory(output_dir)
    records = [{"code": s.get("code"), "name": s.get("name"), "price": s.get("price"), "score": s.get("score")} for s in recommendations]
    memory["last_run"] = run_id
    memory["last_records"] = records
    with open(path, "w", encoding="utf-8") as f: json.dump(memory, f, ensure_ascii=False, indent=2)
    return records

def get_learning_adjustment(stock, indicators, memory, config): return 0.0
