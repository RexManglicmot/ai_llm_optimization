import pandas as pd

def accuracy(y_true, y_pred):
    """
    Fraction correct; ignores None predictions.
    Taking corr
    
    """
    ok = [int(p == t) for p, t in zip(y_pred, y_true) if p is not None and t is not None]
    return sum(ok) / len(ok) if ok else 0.0

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect one row per run with columns:
      ['temp','top_p','acc','n','avg_tokens','latency_s'].
    Group by (temp, top_p) and compute simple means.
    """
    g = df.groupby(["temp", "top_p"], as_index=False).agg(
        acc_mean=("acc", "mean"),
        tokens_mean=("avg_tokens", "mean"),
        latency_mean=("latency_s", "mean"),
        n_total=("n", "sum"),
        runs=("acc", "size"),
    )
    g["acc_per_token"] = g["acc_mean"] / g["tokens_mean"].replace(0, 1)
    return g.sort_values("acc_mean", ascending=False).reset_index(drop=True)

def pick_best(summary_df: pd.DataFrame, prefer: str = "acc_mean") -> pd.Series:
    """Choose the top setting; tie-break by lower tokens."""
    return summary_df.sort_values([prefer, "tokens_mean"], ascending=[False, True]).iloc[0]
