# app/runner.py
from app.prompt import build_prompt_numeric, parse_index_or_letter
import re
import time

def eval_once(rows, hf_client, temperature, top_p, max_new_tokens=2, log_records=False):
    preds, ground_truth, token_counts, records = [], [], [], []  
    t0 = time.perf_counter()

    for i, r in enumerate(rows):
        prompt = build_prompt_numeric(r["question"], r["choices"])
        text = hf_client.generate(prompt, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        first_line = (text or "").splitlines()[0]
        pred = parse_index_or_letter(first_line)

        preds.append(pred)
        ground_truth.append(r["answer"])
        token_counts.append(len(prompt.split()) + len(first_line.split()))

        if log_records:
            records.append({
                "idx": i,
                "temperature": temperature,
                "top_p": top_p,
                "pred": pred,
                "question": r["question"],
                "choices": r["choices"],
                "ground_truth": r["answer"],
                "correct_or_not": int(pred == r["answer"]), #0 is no and 1 is yes
                "first_line": first_line, # first line of the model’s output because output may be "C\nBecause the patient..." → first_line becomes "C" but is going to be "2"
                # optional, can be large, "prompt": prompt
            })

    summary = {
        "acc": sum(int(p == g) for p, g in zip(preds, ground_truth)) / len(rows),
        "avg_tokens": sum(token_counts) / len(token_counts),
        "latency_s": time.perf_counter() - t0,
        "n": len(rows),
    }
    return (summary, records) if log_records else summary
