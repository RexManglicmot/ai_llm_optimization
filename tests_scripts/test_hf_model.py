from app.hf_model import HFClient

hf = HFClient("google/gemma-2-2b-it")

# Greedy (deterministic) — no temp/top_p
print(hf.generate("Say A or B only.\nAnswer:", max_new_tokens=2))

# Sampling — now pass temp/top_p
print(hf.generate("Say A or B only.\nAnswer:", temperature=0.7, top_p=0.9, max_new_tokens=2))
