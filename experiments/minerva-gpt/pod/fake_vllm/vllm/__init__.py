"""A stand-in for vLLM, for smoke_context.py on a machine with no NVIDIA GPU: the real vllm_sample.py
runs against it. Same calls, checked by hand against vLLM 0.32's source. It proves the plumbing, not
vLLM itself; only the crosscheck on the pod can do that.

FAKE_VLLM=hf (default): log-probs from the model through plain Transformers. FAKE_VLLM=fast: made-up
but repeatable numbers, for runs over real prompts too long for a CPU.
Faults, to test the code that handles them:
  FAKE_VLLM_NO_WINDOW=1            ignore the sliding window, as a broken vLLM might
  FAKE_VLLM_LONG_FAULT=x           add x to the log-probs of long prompts only (over twice the window)
  FAKE_VLLM_CACHE_FAULT=1          greedy answers go wrong from the second pass, as a bad prefix cache might
  FAKE_VLLM_NAN=1                  score one answer as NaN
  FAKE_VLLM_REQUIRE_BACKEND=NAME   crash at start unless attention_config asks for this backend
  FAKE_VLLM_CRASH_AFTER=N          crash after N generate() calls
  FAKE_VLLM_LOG=path               record each call: backend, prompts, longest prompt
"""

import hashlib
import json
import os
from types import SimpleNamespace

ANSWERS = ["So I think, like, the main point is the evidence. Yeah.",
           "I think it depends on the sample, like how it was collected.",
           "So yeah, the model is overfitting because the classes are like imbalanced.",
           "Right, the reading says the same thing about the second method.",
           "So the first part is fine.\nProfessor:",
           "Yep, like I said, it comes down to the assumptions we make."]


class SamplingParams:
    """Only the fields vllm_sample.py uses, each checked in vLLM 0.32's sampling_params.py."""

    def __init__(self, n=1, temperature=1.0, top_p=1.0, max_tokens=16, stop_token_ids=None, stop=None, seed=None,
                 prompt_logprobs=None):
        self.n, self.temperature, self.top_p, self.max_tokens = n, temperature, top_p, max_tokens
        self.stop_token_ids, self.stop, self.seed, self.prompt_logprobs = stop_token_ids, stop, seed, prompt_logprobs


class LLM:
    calls = 0

    def __init__(self, model, dtype=None, enable_lora=False, max_lora_rank=None, max_model_len=None, seed=None,
                 gpu_memory_utilization=None, max_num_seqs=None, max_num_batched_tokens=None,
                 enable_prefix_caching=None, attention_config=None):
        """Only the arguments vllm_sample.py uses, each checked in vLLM 0.32's llm.py and arg_utils.py."""
        self.passes = 0
        self.backend = (attention_config or {}).get("backend", "auto")
        required = os.environ.get("FAKE_VLLM_REQUIRE_BACKEND")
        if required and self.backend != required:
            raise RuntimeError(f"fake vLLM: backend {self.backend} does not run here, needs {required}")
        self.mode, self.model = os.environ.get("FAKE_VLLM", "hf"), None
        self.path = model

    def _load(self):
        if self.model is None:
            import common

            self.model = common.load_model(name=self.path, attn="sdpa").eval()
            if os.environ.get("FAKE_VLLM_NO_WINDOW"):
                self.model.config.sliding_window = 10 ** 9
        return self.model

    def _note(self, prompts, seeds):
        path = os.environ.get("FAKE_VLLM_LOG")
        if path:
            with open(path, "a") as f:
                f.write(json.dumps({"backend": self.backend, "prompts": len(prompts),
                                    "longest": max(len(p["prompt_token_ids"]) for p in prompts),
                                    "shortest": min(len(p["prompt_token_ids"]) for p in prompts),
                                    "scoring": self.scoring, "seeds": seeds, "hf_token": bool(os.environ.get("HF_TOKEN"))}) + "\n")
        LLM.calls += 1
        limit = os.environ.get("FAKE_VLLM_CRASH_AFTER")
        if limit and LLM.calls > int(limit):
            raise RuntimeError("fake vLLM: crash on request")

    def _logprobs(self, ids):
        if self.mode == "fast":
            return [None] + [{t: SimpleNamespace(logprob=-1 - hashlib.sha1(f"{len(ids)}:{k}:{t}".encode()).digest()[0] / 64)}
                             for k, t in enumerate(ids[1:], 1)]
        import torch

        model = self._load()
        seq = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            lp = model(input_ids=seq, use_cache=False).logits[0, :-1].float().log_softmax(-1)
        fault = float(os.environ.get("FAKE_VLLM_LONG_FAULT", 0))
        if fault and len(ids) <= 2 * model.config.sliding_window:
            fault = 0
        return [None] + [{t: SimpleNamespace(logprob=lp[k, t].item() + fault)} for k, t in enumerate(ids[1:])]

    def _greedy(self, ids, max_tokens, eos):
        import torch

        model = self._load()
        seq = torch.tensor([ids], device=model.device)
        out = model.generate(seq, max_new_tokens=max_tokens, do_sample=False, eos_token_id=eos, pad_token_id=eos)[0, len(ids):].tolist()
        if os.environ.get("FAKE_VLLM_CACHE_FAULT") and self.passes > 1:
            out = [(t + 1) % model.config.vocab_size for t in out]
        return out

    def generate(self, prompts, params, lora_request=None):
        per_prompt = params if isinstance(params, list) else [params] * len(prompts)
        assert len(per_prompt) == len(prompts), "one SamplingParams per prompt, as vLLM requires"
        self.scoring = per_prompt[0].prompt_logprobs is not None
        self.passes += 1
        self._note(prompts, [q.seed for q in per_prompt])
        out = []
        for p, params in zip(prompts, per_prompt):
            ids = p["prompt_token_ids"]
            if self.scoring:
                lp = self._logprobs(ids)
                if os.environ.get("FAKE_VLLM_NAN"):
                    lp[-1] = {ids[-1]: SimpleNamespace(logprob=float("nan"))}
                out.append(SimpleNamespace(prompt_logprobs=lp, outputs=[]))
                continue
            if params.temperature == 0:
                out.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=self._greedy(ids, params.max_tokens, params.stop_token_ids[0]))]))
                continue
            seed = int(hashlib.sha1(json.dumps(ids).encode()).hexdigest(), 16)
            outputs = []
            for k in range(params.n):
                text = ANSWERS[(seed + k) % len(ANSWERS)]
                stop = next((s for s in (params.stop or []) if s in text), None)
                outputs.append(SimpleNamespace(text=text.split(stop)[0] if stop else text, stop_reason=stop,
                                               finish_reason="length" if (seed + k) % 17 == 0 else "stop"))
            out.append(SimpleNamespace(outputs=outputs))
        return out
