"""Sample answers from a trained adapter to held-out asks, for ranking and evals.

    uv run python generate.py --adapter checkpoints/sft-A --variant A --n 2 --limit 200
    uv run python generate.py --adapter checkpoints/sft-B --variant B --n 2 --retrieve

Reads my examples from data/sft/{split}.jsonl (val by default: never trained on, and kept
apart from test, which stays for the final eval) and writes one row per ask to
data/generations/{variant}-{split}.jsonl:
    {"id", "class_id", "source", "los", "prompt", "real", "variant", "samples": [...]}
`--retrieve` adds my own top-graded train-split answers on the same LOs/HCs to the prompt
(retrieve.py), the "retrieval informs" part of the preference stage.

`--engine hf` (default, any device) uses transformers' generate. `--engine vllm` (the GPU pod)
hands the same token ids to vllm_sample.py in its own environment ($VLLM_PYTHON, default
~/vllm-env/bin/python), which batches sampling far better.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch

import common

SAMPLING = {"do_sample": True, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 256}


def ask_id(row):
    return hashlib.sha1(row["prompt"][0]["content"].encode()).hexdigest()[:12]


def sample(model, tok, prompts, n, batch=4):
    """n sampled answers per prompt (lists of chat messages), batched with left padding."""
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok.pad({"input_ids": [prompt_ids(tok, p) for p in prompts[i:i + batch]]},
                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, num_return_sequences=n, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, **SAMPLING)
        answers = tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [[a.strip() for a in answers[j * n:(j + 1) * n]] for j in range(len(enc["input_ids"]))]
    return out


def prompt_ids(tok, prompt):
    """The exact token ids both engines sample from: the chat template, generation prompt on."""
    text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


def sample_vllm(adapter, tok, prompts, n):
    python = os.environ.get("VLLM_PYTHON", str(Path.home() / "vllm-env/bin/python"))
    with tempfile.TemporaryDirectory() as tmp:
        job, out = Path(tmp) / "job.json", Path(tmp) / "out.json"
        job.write_text(json.dumps({
            "adapter": str(Path(adapter).resolve()), "prompts": [prompt_ids(tok, p) for p in prompts],
            "n": n, "temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"],
            "max_new_tokens": SAMPLING["max_new_tokens"], "eos_token_id": tok.eos_token_id}))
        subprocess.run([python, str(Path(__file__).with_name("vllm_sample.py")), str(job), str(out)],
                       check=True)
        return json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--variant", choices=["A", "B"], required=True)
    ap.add_argument("--data", type=Path, default=common.DATA)
    ap.add_argument("--split", default="val")
    ap.add_argument("--source", default="class_turn", help="class_turn | pcw | poll | all")
    ap.add_argument("--n", type=int, default=2, help="answers per ask")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retrieve", action="store_true")
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [r for r in common.read_jsonl(args.data / f"sft/{args.split}.jsonl")
            if args.source == "all" or r["source"] == args.source][:args.limit]
    prompts = [r["prompt"] for r in rows]
    if args.retrieve:
        from retrieve import Retriever

        retriever = Retriever(args.data)
        prompts = [retriever.augment(r) for r in rows]

    tok = common.load_tokenizer()
    if args.engine == "vllm":
        samples = sample_vllm(args.adapter, tok, prompts, args.n)
    else:
        model = common.load_model(args.adapter)
        model.eval()
        samples = sample(model, tok, prompts, args.n)

    out = args.out or args.data / f"generations/{args.variant}-{args.split}.jsonl"
    common.write_jsonl(out, [
        {"id": ask_id(r), "class_id": r["class_id"], "source": r["source"], "los": r["los"],
         "prompt": p, "real": r["completion"][0]["content"], "variant": args.variant, "samples": s}
        for r, p, s in zip(rows, prompts, samples, strict=True)])
    print(f"{len(rows)} asks x {args.n} answers -> {out}")


if __name__ == "__main__":
    main()
