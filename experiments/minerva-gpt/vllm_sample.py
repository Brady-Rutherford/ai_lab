"""Sampling with vLLM on the GPU pod; generate.py --engine vllm runs this for you.

vLLM pins its own torch and transformers (0.32: torch 2.13, transformers < 5.20), which the
training environment (torch 2.14, transformers 5.21) can't share. So vLLM lives in its own
environment, ~/vllm-env, and this script talks to generate.py through two JSON files. Prompts
arrive already rendered to token ids by the training tokenizer, so vLLM sees exactly what the
HF path would.

    ~/vllm-env/bin/python vllm_sample.py JOB.json OUT.json

JOB.json: {"adapter", "prompts": [[token ids]...], "n", "temperature", "top_p",
"max_new_tokens", "eos_token_id"}. OUT.json: per prompt, n [answer, finished] pairs; finished
is False when the answer hit max_new_tokens instead of ending.

Scoring instead (eval_context.py): JOB.json {"adapter", "score": [{"prompt": ids, "answer": ids}...]}
-> OUT.json: per item, the mean log-probability of the answer's tokens after its prompt (the
loss eval_compare.py reports, negated). Optional in both: "max_model_len" (default 4096).
"""

import json
import os
import sys

# vLLM's FlashInfer sampler JIT-compiles its top-k/top-p kernel on first use, which needs ninja
# and a CUDA toolkit matching vLLM's torch (13; Lambda images ship nvcc 12.8). PyTorch's
# sampler is the reference implementation of the same sampling, with nothing to compile; for a
# 1B model sampling is a sliver of the time.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402

import common  # standard library only; sets HF offline mode, so the model loads from the cache


def main(job_path, out_path):
    job = json.load(open(job_path))
    merged = common.is_merged(job["adapter"])  # SFT + DPO folded in: vLLM can't stack adapters
    # Throughput settings, none of which change what is sampled: most of the GPU for the KV
    # cache, more sequences in flight, bigger prefill batches, and prefix caching, since many
    # asks from one class share its long context block.
    scoring = "score" in job
    llm = LLM(job["adapter"] if merged else common.BASE_MODEL, dtype="bfloat16", enable_lora=not merged,
              max_lora_rank=common.LORA["r"], max_model_len=int(job.get("max_model_len", 4096)), seed=0,
              # scoring keeps a log-prob row per prompt token of each prefill chunk: smaller chunks
              # and a little more free memory than sampling needs
              gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.85" if scoring else "0.92")),
              max_num_seqs=int(os.environ.get("VLLM_MAX_SEQS", "512")),
              max_num_batched_tokens=int(os.environ.get("VLLM_BATCHED_TOKENS", "4096" if scoring else "16384")),
              enable_prefix_caching=not scoring)  # prompt log-probs are computed in full
    lora = None if merged else LoRARequest("adapter", 1, job["adapter"])
    if scoring:
        # Chunks of 64: vLLM returns a log-prob entry for every prompt position, and full-class
        # prompts run to ~25k tokens, so the results are reduced to one number per item as they come.
        items, means, chunk = job["score"], [], 64
        for i in range(0, len(items), chunk):
            part = items[i:i + chunk]
            outputs = llm.generate([TokensPrompt(prompt_token_ids=it["prompt"] + it["answer"]) for it in part],
                                   SamplingParams(max_tokens=1, prompt_logprobs=0), lora_request=lora)
            for it, o in zip(part, outputs):
                ids, start = it["prompt"] + it["answer"], len(it["prompt"])
                lp = [o.prompt_logprobs[k][ids[k]].logprob for k in range(start, len(ids))]
                means.append(sum(lp) / len(lp))
        json.dump(means, open(out_path, "w"))
        return
    params = SamplingParams(n=job["n"], temperature=job["temperature"], top_p=job["top_p"],
                            max_tokens=job["max_new_tokens"], stop_token_ids=[job["eos_token_id"]],
                            seed=0)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in job["prompts"]],
                           params, lora_request=lora)
    json.dump([[[c.text.strip(), c.finish_reason == "stop"] for c in o.outputs] for o in outputs],
              open(out_path, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
