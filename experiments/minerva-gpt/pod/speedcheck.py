"""On the POD: prove the fast path changes speed, not results, before any real run uses it.

    cd ~/minerva-gpt && uv run python pod/speedcheck.py

Fast path = FlashAttention 3 (prebuilt kernel) + packing + Liger kernels. Two checks:
1. Isolation. Packing puts several examples in one row, and only FlashAttention's
   variable-length mode (driven by position_ids that restart at 0 for each example) keeps them
   from attending to each other. Each example's log-probs packed behind another must match its
   log-probs alone. The same packed row without the restarts (one long sequence) must not match:
   that negative control shows the test can see contamination at all.
2. Equivalence. One epoch of SFT on the same 640 examples, verified path vs fast path: the
   validation losses must agree within 2%, and the fast path must actually be faster.
On PASS it writes the flags to ~/fast_flags, which pod/run.sh picks up; on FAIL it deletes them.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

FAST = ["--attn", "flash_attention_3", "--packing", "--liger"]
FLAGS_FILE = Path.home() / "fast_flags"
ISOLATED_MAX = 0.02    # mean |Δ log p| per token when packed vs alone (bf16 noise is far below)
CONTAMINATED_MIN = 0.1  # the negative control must move log-probs at least this much
LOSS_TOLERANCE = 0.02
SUBSET = 640           # 20 optimizer steps at the default effective batch of 32


def token_logprobs(model, ids, positions=None):
    ids_t = torch.tensor([ids], device=model.device)
    kwargs = {"position_ids": torch.tensor([positions], device=model.device)} if positions else {}
    with torch.no_grad():
        logits = model(input_ids=ids_t, **kwargs).logits[0].float()
    return logits[:-1].log_softmax(-1).gather(-1, ids_t[0, 1:, None])[:, 0]


def isolation():
    tok, model = common.load_tokenizer(), common.load_model(attn="flash_attention_3")
    model.eval()
    a, b = ([tok(tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False),
                 add_special_tokens=False).input_ids[:512]
             for r in common.read_jsonl(common.DATA / "sft/val.jsonl")[:2]])
    alone = token_logprobs(model, b)
    packed = token_logprobs(model, a + b, list(range(len(a))) + list(range(len(b))))[len(a):]
    leaky = token_logprobs(model, a + b)[len(a):]
    isolated = (packed - alone).abs().mean().item()
    control = (leaky - alone).abs().mean().item()
    print(f"isolation: packed vs alone {isolated:.4f} (need <= {ISOLATED_MAX}); "
          f"negative control {control:.4f} (need >= {CONTAMINATED_MIN})")
    del model
    torch.cuda.empty_cache()
    return isolated <= ISOLATED_MAX and control >= CONTAMINATED_MIN


def sft_run(extra, out):
    cmd = [sys.executable, "train_sft.py", "--variant", "A", "--limit", str(SUBSET), "--batch", "8",
           "--no-checkpointing", "--out", str(out), *extra]
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=common.ROOT)
    if run.returncode != 0:
        raise RuntimeError(f"train_sft.py {' '.join(extra) or '(verified)'} failed: {run.stderr.strip()[-500:]}")
    log = run.stdout
    loss = float(re.search(r"val loss: ([\d.]+)", log).group(1))
    minutes = float(re.search(r"in ([\d.]+) min", log).group(1))
    return loss, minutes


def equivalence():
    with tempfile.TemporaryDirectory() as tmp:
        base_loss, base_min = sft_run([], Path(tmp) / "base")
        fast_loss, fast_min = sft_run(FAST, Path(tmp) / "fast")
    rel = abs(fast_loss - base_loss) / base_loss
    print(f"equivalence: val loss {base_loss:.4f} verified vs {fast_loss:.4f} fast ({rel:.1%}, need "
          f"<= {LOSS_TOLERANCE:.0%}); {base_min:.1f} vs {fast_min:.1f} min ({base_min / fast_min:.1f}x)")
    return rel <= LOSS_TOLERANCE and fast_min < base_min


def main():
    try:
        ok = isolation() and equivalence()
    except Exception as e:  # a missing kernel or an incompatible build: stay on the verified path
        print(f"fast path unavailable: {type(e).__name__}: {e}")
        ok = False
    if ok:
        FLAGS_FILE.write_text(" ".join(FAST))
        print(f"PASS: pod/run.sh will train with: {' '.join(FAST)}")
    else:
        FLAGS_FILE.unlink(missing_ok=True)
        print("FAIL: pod/run.sh stays on the verified sdpa path")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
