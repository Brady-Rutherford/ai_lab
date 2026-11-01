"""Ask what a professor would ask; the model answers the way I would in class.

    uv run --extra mac python chat.py                    # 8-bit copy of model B (SFT + DPO), with MLX
    uv run --extra mac python chat.py --class climate    # start in a class (see below)
    uv run python chat.py --full                         # full bf16 model, transformers (needs a GPU
                                                         # with ~16 GB free: the reference, not for the Mac)

Each answer is built from a prompt in the training format: "Answer as: Brady", the class header
(course, session, focused LOs/HCs), that class's readings and my pre-class work, then
"Professor: <your question>" and "What do you say?". Questions are independent, as in training.
At the prompt:
    /class WORDS   list the classes whose course, title or LOs contain all the words
    /class N       use match N from the last list (one match is picked straight away)
    /noclass       no class context
    /quit
Class context comes from data/sft/ (local only), so on a machine without it, ask without one.
The model comes from my private HF repo on first use (needs the HF login), then from the local
cache with no network.
"""

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "0")  # common.py defaults to offline; the first run downloads

import common
from generate import SAMPLING

REPO = "bradyrutherford/minerva-olmo3-7b-{variant}"


def classes():
    """class id -> (header, context) from my class-discussion examples: the header without the
    activity line, the context without the poll that happened to be on screen."""
    out = {}
    for split in ("train", "val", "test"):
        path = common.DATA / "sft" / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in common.read_jsonl(path):
            if row["source"] != "class_turn" or row["class_id"] in out:
                continue
            text = row["prompt"][0]["content"]
            head = text.split("\n\n")[0]
            end = text.find("\n\nProfessor:")
            ctx = text[len(head) + 2:end] if end > len(head) else ""
            ctx = "" if ctx.startswith("On screen (poll):") else ctx.split("\nOn screen (poll):")[0]
            head = "\n".join(l for l in head.split("\n") if not l.startswith("Activity:"))
            out[row["class_id"]] = (head, ctx)
    return out


def prompt(question, klass):
    head, ctx = klass or (f"{common.TAG_ME}\nClass discussion", "")
    body = f"Professor: {question}\n\nWhat do you say?"
    return [{"role": "user", "content": "\n\n".join(x for x in (head, ctx, body) if x)}]


def local(repo):
    """The model folder: as given, or from the HF cache, downloading only if it isn't there."""
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        print(f"first use: downloading {repo} from Hugging Face", file=sys.stderr)
        return snapshot_download(repo)


def mlx_engine(path):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(path)
    hf = tok._tokenizer  # the plain HF tokenizer: the exact token ids training used
    sampler = make_sampler(temp=SAMPLING["temperature"], top_p=SAMPLING["top_p"])

    def answer(messages):
        text = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = hf(text, add_special_tokens=False).input_ids
        for r in stream_generate(model, tok, ids, max_tokens=SAMPLING["max_new_tokens"], sampler=sampler):
            print(r.text, end="", flush=True)
        print()
    return answer


def hf_engine(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def answer(messages):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        model.generate(ids, streamer=streamer, eos_token_id=tok.eos_token_id, **SAMPLING)
    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B", choices=["A", "B"])
    ap.add_argument("--full", action="store_true", help="the full bf16 model with transformers")
    ap.add_argument("--model", help="a repo id or local folder instead of the default")
    ap.add_argument("--class", dest="klass", metavar="WORDS", help="start in this class")
    args = ap.parse_args()
    repo = args.model or REPO.format(variant=args.variant) + ("" if args.full else "-mlx-8bit")
    answer = (hf_engine if args.full else mlx_engine)(local(repo))
    known, matches, klass = classes(), [], None

    def pick(words):
        nonlocal matches, klass
        if words.isdigit() and 0 < int(words) <= len(matches):
            klass = known[matches[int(words) - 1]]
        else:
            terms = words.lower().split()
            matches = [c for c, (head, _) in known.items() if all(t in head.lower() for t in terms)]
            for i, c in enumerate(matches[:15], 1):
                print(f"  {i:2d}. {known[c][0].split(chr(10))[1]}")
            if len(matches) > 15:
                print(f"  ... {len(matches) - 15} more: add words to narrow it")
            if len(matches) != 1:
                print("  no class matches" if not matches else "  /class N to pick one")
                return
            klass = known[matches[0]]
        print("  class:", klass[0].split("\n")[1])

    if args.klass:
        pick(args.klass)
    print(f"{repo}: ask what a professor would ask (/class WORDS, /noclass, /quit)")
    while True:
        try:
            line = input("\nProfessor> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/class"):
            pick(line[len("/class"):].strip())
        elif line == "/noclass":
            klass = None
        elif line:
            answer(prompt(line, klass))


if __name__ == "__main__":
    main()
