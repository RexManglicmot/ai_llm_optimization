# plots/plot_val_heatmap.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "/Users/Rex/vscode/ai_llm_optimization/results/google_gemma-2-9b-it/val_summary_google_gemma-2-9b-it.csv"
df = pd.read_csv(path)

piv = df.pivot(index="top_p", columns="temp", values="acc_mean").sort_index().sort_index(axis=1)
plt.imshow(piv.values, aspect="auto")
plt.xticks(range(len(piv.columns)), piv.columns)
plt.yticks(range(len(piv.index)), piv.index)
plt.colorbar(label="Accuracy")
plt.xlabel("temperature")
plt.ylabel("top_p")
plt.title("Gemma-2-9B (val): accuracy heatmap")
plt.tight_layout()
plt.savefig("results/google_gemma-2-9b-it/val_heatmap.png", dpi=150)
