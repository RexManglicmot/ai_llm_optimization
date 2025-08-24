# plots/plot_val_pareto.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/Rex/vscode/ai_llm_optimization/results/google_gemma-2-9b-it/val_summary_google_gemma-2-9b-it.csv")
plt.scatter(df["tokens_mean"], df["acc_mean"])
for _, r in df.iterrows():
    lbl = f"T{r['temp']}/P{r['top_p']}"
    plt.text(r["tokens_mean"], r["acc_mean"], lbl, fontsize=8)

plt.xlabel("Avg tokens")
plt.ylabel("Accuracy")
plt.title("Gemma-2-9B (val): accuracy vs tokens")
plt.tight_layout()
plt.savefig("results/google_gemma-2-9b-it/val_acc_vs_tokens.png", dpi=150)
