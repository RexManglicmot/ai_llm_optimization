# app/hf_model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class HFClient:
    def __init__(self, model_id: str = "google/gemma-2-2b-it", fourbit: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if fourbit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb, device_map="auto", low_cpu_mem_usage=True
            )
        else:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
            )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, return_full_text=False)

    # def generate(self, prompt, temperature=None, top_p=None, max_new_tokens=2):
    #     kwargs = {
    #         "max_new_tokens": max_new_tokens,
    #         "pad_token_id": self.tokenizer.eos_token_id,
    #         "return_full_text": False,
    #     }
    #     # turn sampling on only if needed
    #     sample = (
    #         (temperature is not None and float(temperature) > 0.0) or
    #         (top_p is not None and float(top_p) < 1.0)
    #     )
    #     if sample:
    #         kwargs["do_sample"] = True
    #         if temperature is not None:
    #             kwargs["temperature"] = float(temperature)
    #         if top_p is not None:
    #             kwargs["top_p"] = float(top_p)
    #     else:
    #         kwargs["do_sample"] = False  # greedy; don't send temp/top_p

    #     return self.pipe(prompt, **kwargs)[0]["generated_text"]

    # app/hf_model.py
    def generate(self, prompt, temperature=None, top_p=None, max_new_tokens=2):
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_full_text": False,
        }

        # decide if we are sampling
        t = None if temperature is None else float(temperature)
        p = None if top_p is None else float(top_p)
        wants_sampling = (t is not None and t > 0.0) or (p is not None and p < 1.0)

        kwargs["do_sample"] = wants_sampling

        if wants_sampling:
            # only pass knobs that actually apply
            if t is not None and t > 0.0:
                kwargs["temperature"] = t
            if p is not None and p < 1.0:
                kwargs["top_p"] = p
            # Note: if t==0.0 and p<1.0, we sample with default temperature (OK)
        # else: greedy — do NOT include temperature/top_p

        return self.pipe(prompt, **kwargs)[0]["generated_text"]

