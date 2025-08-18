# tests_scripts/test_metrics_min.py
import os
import pandas as pd
from app.metrics import aggregate, pick_best, accuracy

# 🔧 HARD-CODED OUTPUT PATH (change if you want a different file)
OUT_PATH = "/Users/Rex/vscode/ai_llm_optimization/metrics_result/test_metrics_results.csv"

def main():
    # --- Fake runs: exactly the columns your aggregate() expects ---
    df = pd.DataFrame([
        {"temp": 0.0, "top_p": 1.0, "acc": 0.70, "n": 100, "avg_tokens": 50, "latency_s": 1.0},
        {"temp": 0.0, "top_p": 1.0, "acc": 0.72, "n": 120, "avg_tokens": 48, "latency_s": 0.9},
        {"temp": 0.5, "top_p": 0.9, "acc": 0.66, "n": 100, "avg_tokens": 60, "latency_s": 1.2},
        {"temp": 0.5, "top_p": 0.9, "acc": 0.64, "n": 110, "avg_tokens": 61, "latency_s": 1.1},
    ])

    # Quick sanity check of accuracy()
    assert abs(accuracy([0,1,2,3], [0,2,2,3]) - 0.75) < 1e-9, "accuracy() sanity check failed"

    # Aggregate & pick best
    summary = aggregate(df)
    best = pick_best(summary)

    print("\nSUMMARY:")
    print(summary[["temp","top_p","acc_mean","tokens_mean","latency_mean","n_total","runs","acc_per_token"]]
          .to_string(index=False))
    print("\nBEST SETTING (by acc_mean, tie-break tokens):")
    print(best.to_string())

    # --- Write CSV to the hard-coded path ---
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)
    print(f"\nWrote summary to: {OUT_PATH}")

if __name__ == "__main__":
    main()


# Run: Python3 -m tests_scripts.test_metrics