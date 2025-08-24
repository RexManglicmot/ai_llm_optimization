# scripts/run_test.py
import argparse
import os
import pandas as pd

from app.hf_model import HFClient
from app.runner import eval_once
from app.data_loader import load_split_local


def main():
    ap = argparse.ArgumentParser(description="Run final test with a fixed decoding policy.")
    ap.add_argument("--test-path", required=True, help="Path to test parquet/csv (cleaned).")
    ap.add_argument("--model", default="google/gemma-2-2b-it", help="HF model id or endpoint URL.")
    ap.add_argument("--temperature", type=float, required=True, help="Decoding temperature.")
    ap.add_argument("--top-p", type=float, required=True, help="Top-p nucleus sampling.")
    ap.add_argument("--max-new", type=int, default=2, help="Max new tokens to generate.")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on #items (0 = all).")
    ap.add_argument("--out", default="results/test/test_final_google_gemma-2-9b-it.csv", help="Where to append the test summary row.")
    args = ap.parse_args()

    # Load data
    rows = load_split_local(args.test_path)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    # One model instance reused for all prompts
    hf = HFClient(args.model)

    # Evaluate once with the fixed policy
    res = eval_once(
        rows=rows,
        hf_client=hf,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new,
    )

    # Prepare a single summary row
    record = {
        "model": args.model,
        "temp": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new,
        **res,  # acc, avg_tokens, latency_s, n
    }

    # Ensure parent directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Append (create if not exists)
    write_header = not os.path.exists(args.out)
    pd.DataFrame([record]).to_csv(args.out, mode="a", header=write_header, index=False)

    # Console report
    print("\nFinal TEST results")
    print("------------------")
    print(f"Model        : {args.model}")
    print(f"Policy       : temperature={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new}")
    print(f"Items (n)    : {res['n']}")
    print(f"Accuracy     : {res['acc']:.4f}")
    print(f"Avg tokens   : {res['avg_tokens']:.2f}")
    print(f"Total latency: {res['latency_s']:.2f}s")
    print(f"\nAppended to  : {args.out}")


if __name__ == "__main__":
    main()

"""
Best model was google_gemma-2-9bit
temp = 0
top-p = 0.9
df_test has 14042 rows

ISSUE: was using the val rows of 1531, went back to data cleaning and changed it.

Run python -m scripts.run_test \
  --model google/gemma-2-9b-it \
  --test-path "/Users/Rex/vscode/ai_llm_optimization/data/clean/df_test_cleaned.parquet" \
  --temperature 0.0 --top-p 0.9 --max-new 1

Start 2:38
End....520....nothing showing up   

It worked with 10, now lets try with 14042
Start 5:33pm
End 630 pm
"""