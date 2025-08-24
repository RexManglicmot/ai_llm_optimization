# app/hf_model.py
import os
from typing import Optional
from transformers import pipeline, AutoTokenizer
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError


__all__ = ["HFClient"]

class HFClient:
    """
    Minimal unified client:
      - backend="local"     → Transformers pipeline on your machine
      - backend="inference" → Hugging Face Inference (hosted, chat endpoint)

    Env (for inference):
      HF_BACKEND=inference
      HUGGINGFACEHUB_API_TOKEN=hf_xxx
      # Optional:
      # HF_INFERENCE_PROVIDER=hf-inference
    """
    def __init__(self, model_id: str, backend: Optional[str] = None):
        self.model_id = model_id
        self.backend = (backend or os.getenv("HF_BACKEND") or "local").lower()

        if self.backend == "inference":
            token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not token:
                raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is required for backend='inference'.")
            provider = os.getenv("HF_INFERENCE_PROVIDER")  # optional
            self.client = InferenceClient(model=model_id, token=token, provider=provider)
            self.pipe = None
            self.tokenizer = None
        else:
            # Local transformers pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.pipe = pipeline(
                task="text-generation",
                model=model_id,
                tokenizer=self.tokenizer,
                device_map="auto",
                return_full_text=False,  # only the generated tail
            )
            self.client = None

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: int = 2,
    ) -> str:
        """
        Generate a short completion for `prompt`.
        - Greedy if NOT sampling (temp<=0 and top_p>=1.0)
        - Sampling if (temp>0) OR (top_p<1.0)
        """
        t = None if temperature is None else float(temperature)
        p = None if top_p is None else float(top_p)
        wants_sampling = ((t is not None and t > 0.0) or (p is not None and p < 1.0))

        if self.backend == "inference":
            # Use chat endpoint (serverless often exposes instruct models as chat)
            cc = {
                "max_tokens": int(max_new_tokens),  # chat API name
                "temperature": (t if wants_sampling and t is not None else 0.0),
                "top_p": (p if wants_sampling and p is not None else 1.0),
            }

            last_err = None
            for attempt in range(5):  # 5 tries with backoff: ~1s,2s,4s,8s,16s
                try:
                    resp = self.client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        **cc,
                    )
                    return resp.choices[0].message.content.strip()
                except HfHubHTTPError as e:
                    # Retry only on server errors (>=500)
                    code = getattr(e.response, "status_code", None)
                    if code and code >= 500:
                        time.sleep(1 * (2 ** attempt))
                        last_err = e
                        continue
                    raise
                except Exception as e:
                    # network hiccup → retry
                    time.sleep(1 * (2 ** attempt))
                    last_err = e
                    continue
            # if we get here, all retries failed
            raise last_err

            resp = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **cc,
            )
            return resp.choices[0].message.content.strip()

        # Local transformers (text-generation pipeline)
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_full_text": False,
            "do_sample": wants_sampling,
        }
        if wants_sampling:
            if t is not None and t > 0.0:
                kwargs["temperature"] = t
            if p is not None and p < 1.0:
                kwargs["top_p"] = p
        out = self.pipe(prompt, **kwargs)
        return out[0]["generated_text"].strip()
