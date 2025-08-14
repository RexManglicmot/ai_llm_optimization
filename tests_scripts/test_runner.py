
from app.data_loader import load_split_local
from app.hf_model import HFClient
from app.runner import eval_once
import pandas as pd
import os

rows = load_split_local("/Users/Rex/vscode/ai_llm_optimization/data/clean/df_val_cleaned.parquet")[:5]
hf = HFClient("google/gemma-2-2b-it")
summary, records = eval_once(rows, hf, temperature=0.0, top_p=1.0, max_new_tokens=2, log_records=True)
print(summary)

os.makedirs("results", exist_ok=True)
pd.DataFrame(records).to_csv("/Users/Rex/vscode/ai_llm_optimization/runner_results/eval_once.csv", index=False)

# Run: Python3 -m tests_scripts.test_runner

