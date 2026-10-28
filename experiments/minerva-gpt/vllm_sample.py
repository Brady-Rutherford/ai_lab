"""Sampling with vLLM on the GPU pod; generate.py --engine vllm runs this for you.

vLLM pins its own torch and transformers (0.32: torch 2.13, transformers < 5.20), which the
training environment (torch 2.14, transformers 5.21) can't share. So vLLM lives in its own
environment, ~/vllm-env, and this script talks to generate.py through two JSON files. Prompts
arrive already rendered to token ids by the training tokenizer, so vLLM sees exactly what the
HF path would.

    ~/vllm-env/bin/python vllm_sample.py JOB.json OUT.json

JOB.json: {"adapter", "prompts": [[token ids]...], "n", "temperature", "top_p",
"max_new_tokens", "eos_token_id"}. OUT.json: one list of n answers per prompt.
"""

import json
import sys

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

import common  # standard library only; sets HF offline mode, so the model loads from the cache


def main(job_path, out_path):
    job = json.load(open(job_path))
    llm = LLM(common.BASE_MODEL, dtype="bfloat16", enable_lora=True,
              max_lora_rank=common.LORA["r"], max_model_len=4096, seed=0)
    params = SamplingParams(n=job["n"], temperature=job["temperature"], top_p=job["top_p"],
                            max_tokens=job["max_new_tokens"], stop_token_ids=[job["eos_token_id"]],
                            seed=0)
    lora = LoRARequest("adapter", 1, job["adapter"])
    outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in job["prompts"]],
                           params, lora_request=lora)
    json.dump([[c.text.strip() for c in o.outputs] for o in outputs], open(out_path, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
