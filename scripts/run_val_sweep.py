# scripts/run_val_sweep.py
import argparse, os, pandas as pd
from app.hf_model import HFClient
from app.runner import eval_once
from app.metrics import aggregate, pick_best
from app.data_loader import load_split_local  
import time #added after google/gemma-2-9b-it

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-path", required=True, help="Path to validation parquet/csv you already use")
    ap.add_argument("--out-raw", default="results/google_gemma-2-2b-it/val_raw_google_gemma-2-2b-it.csv")
    #orginal
    #default="results/val_raw.csv"

    ap.add_argument("--out-summary", default="results/google_gemma-2-2b-it/val_summary_google_gemma-2-2b-it.csv")
    #orginal
    #default="results/val_summary.csv"

    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--max-new", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="Optional: cap #items for a quick pass (0 = all)")
    args = ap.parse_args()

    rows = load_split_local(args.val_path)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    temps  = [0.0, 0.3, 0.7]
    top_ps = [0.9, 1.0]

    hf = HFClient(args.model)  # load once
    records = []
    for t in temps:
        for p in top_ps:
            res = eval_once(rows, hf, temperature=t, top_p=p, max_new_tokens=args.max_new)
            records.append({"temp": t, "top_p": p, **res})

    os.makedirs(os.path.dirname(args.out_raw) or ".", exist_ok=True)
    pd.DataFrame(records).to_csv(args.out_raw, index=False)

    summary = aggregate(pd.DataFrame(records))
    summary.to_csv(args.out_summary, index=False)

    best = pick_best(summary)
    print("\nLeaderboard (top):")
    print(summary.head(10).to_string(index=False))
    print("\nRecommended:", best.to_dict())
    print(f"\nWrote {args.out_raw} and {args.out_summary}")

if __name__ == "__main__":
    main()

"""
Run python -m scripts.run_val_sweep \
  --val-path "/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet" \
  --limit 150

It will give a csv  in results/val_summarycs.v and in the pick_best() is printed.

# Start 8:28 pm
# End

OVERHEAT
Need to change limit and put max_new =1 (only 1 token)

Next day8/23
Bought the monthly HF sub

try again

Run python -m scripts.run_val_sweep \
  --val-path "/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet" \
  --limit 150 --max-new 1

Start 10:35am
End   

Start 10:58am
End 11:05am
Results are pretty bad acc_mean is .50!!

changed it to Qwen/Qwen2.5-7B-Instruct
Start 11:16am
Error 11:25...no folder...because of the "/" in the model..fixed it in the script


Also changed the iteratios to 400 like it was before

Run python -m scripts.run_val_sweep \
  --model Qwen/Qwen2.5-7B-Instruct \
  --val-path "/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet" \
  --limit 400  --max-new 1

Start 11:37am
End 11:52am

67% mean_acc

Try different model
meta-llama/Meta-Llama-3-8B-Instruct

Run python -m scripts.run_val_sweep \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --val-path "/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet" \
  --limit 400 --max-new 1


Start 12:11pm 
End 12:47 pm

BAD!! acc_mean is 52%

Lets try google/gemma-2-9b-it

python -m scripts.run_val_sweep \
  --model google/gemma-2-9b-it \
  --val-path "/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet" \
  --limit 400 --max-new 1

Start 1256pm
Error 114 pm... about importing time??? 

still importing old hf_model_2

(venv_ai_llm_optimization) Rex@MacBook-Pro-376 ai_llm_optimization % grep -R "from app\.hf_model" -n .
grep -R "HFClient" -n scripts app tests_scripts
./scripts/run_val_sweep.py:3:from app.hf_model_old_2 import HFClient
./tests_scripts/test_hf_model.py:1:from app.hf_model_old_2 import HFClient
./tests_scripts/test_runner.py:3:from app.hf_model_old_2 import HFClient
scripts/run_val_sweep.py:3:from app.hf_model_old_2 import HFClient
scripts/run_val_sweep.py:32:    hf = HFClient(args.model)  # load once
Binary file scripts/__pycache__/run_val_sweep.cpython-312.pyc matches
app/hf_model.py:10:__all__ = ["HFClient"]
app/hf_model.py:13:class HFClient:
Binary file app/__pycache__/hf_model_old_2.cpython-312.pyc matches
Binary file app/__pycache__/hf_model.cpython-312.pyc matches
tests_scripts/test_hf_model.py:1:from app.hf_model_old_2 import HFClient
tests_scripts/test_hf_model.py:3:hf = HFClient("google/gemma-2-2b-it")
Binary file tests_scripts/__pycache__/test_runner.cpython-312.pyc matches
Binary file tests_scripts/__pycache__/test_hf_model.cpython-312.pyc matches
tests_scripts/test_runner.py:3:from app.hf_model_old_2 import HFClient
tests_scripts/test_runner.py:9:hf = HFClient("google/gemma-2-2b-it")
(venv_ai_llm_optimization) Rex@MacBook-Pro-376 ai_llm_optimization % sed -i '' 's/from app\.hf_model_old_2 import HFClient/from app.hf_model import HFClient/' \
  scripts/run_val_sweep.py \
  tests_scripts/test_hf_model.py \
  tests_scripts/test_runner.py
(venv_ai_llm_optimization) Rex@MacBook-Pro-376 ai_llm_optimization % find . -name "__pycache__" -type d -exec rm -rf {} +
(venv_ai_llm_optimization) Rex@MacBook-Pro-376 ai_llm_optimization % grep -R "hf_model_old" -n . || echo "OK: no references"
OK: no references
(venv_ai_llm_optimization) Rex@MacBook-Pro-376 ai_llm_optimization % python - <<'PY'
import importlib, inspect
from app.hf_model import HFClient
mod = importlib.import_module(HFClient.__module__)
print("HFClient module:", HFClient.__module__)
print("Module file    :", mod.__file__)
print("Source file    :", inspect.getsourcefile(HFClient))
PY

HFClient module: app.hf_model
Module file    : /Users/Rex/vscode/ai_llm_optimization/app/hf_model.py
Source file    : /Users/Rex/vscode/ai_llm_optimization/app/hf_model.py
(venv_ai_llm_optimization) Rex@MacBook-Pro-376 ai_llm_optimization % set -a; source .env; set +a
(venv_ai_llm_optimization) Rex@MacBook-Pro-376 ai_llm_optimization % 

hopefully fixed

Start 1:25 pm
End 143 pm
71% accurancy

lets go with the 27B

Run python -m scripts.run_val_sweep \
  --model google/gemma-2-27b-it \
  --val-path "/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet" \
  --limit 400 --max-new 1

Start 
Model not available yet.

Lets try the other meta-llama 3B Instruct
Run python -m scripts.run_val_sweep \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --val-path "/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet" \
  --limit 400 --max-new 1

Start 745pm
End 827 pm

Qwen2.5-7B-Instruct
python -m scripts.run_val_sweep \
  --model Qwen/Qwen2.5-3B-Instruct \
  --val-path "/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet" \
  --limit 400 --max-new 1

Doesnt exist


python -m scripts.run_val_sweep \
  --model google/gemma-2-2b-it \
  --val-path "/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet" \
  --limit 400 --max-new 1

Start 904 pm
End 923  
"""
