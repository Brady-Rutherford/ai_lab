"""On the POD, after setup.sh: check everything a run needs, before any GPU time is spent on it.

    cd ~/minerva-gpt && uv run python pod/preflight.py

Must print "all passed". Then, to see the whole pipeline work on this GPU with fake data
(a few minutes): GEN_ENGINE=vllm sh smoke_test.sh
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

VLLM_PYTHON = os.environ.get("VLLM_PYTHON", str(Path.home() / "vllm-env/bin/python"))


def check_gpu():
    import torch

    assert torch.cuda.is_available(), "torch sees no CUDA GPU"
    name, mem = torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9
    assert torch.cuda.is_bf16_supported(), f"{name} has no bfloat16"
    return f"{name}, {mem:.0f} GB, torch {torch.__version__}"


def check_model():
    tok = common.load_tokenizer()
    from transformers import AutoConfig

    AutoConfig.from_pretrained(common.BASE_MODEL)
    return f"{common.BASE_MODEL} loads offline; chat template {'set' if tok.chat_template else 'MISSING'}"


def check_data():
    counts = {p.parent.name + "/" + p.name: sum(1 for _ in open(p))
              for p in sorted(common.DATA.glob("sft*/*.jsonl"))}
    assert counts.get("sft/train.jsonl"), "no data/sft/train.jsonl: run pod/push.sh on the Mac"
    return ", ".join(f"{k} {v}" for k, v in counts.items())


def check_no_secrets():
    found = [str(p) for p in Path.home().rglob(".env") if "site-packages" not in str(p)]
    assert not found, f"secrets files on the pod: {found}"
    return "no .env files on the pod"


def check_vllm():
    out = subprocess.run([VLLM_PYTHON, "-c", "import vllm, torch; print(vllm.__version__, torch.__version__)"],
                         capture_output=True, text=True)
    assert out.returncode == 0, f"vLLM environment broken: {out.stderr.strip()[-300:]}"
    return "vllm {} (torch {}) in its own environment".format(*out.stdout.split())


def check_fast_path():
    from kernels import get_kernel

    get_kernel("kernels-community/vllm-flash-attn3", version=1)  # offline: from setup's cache
    import liger_kernel  # noqa: F401
    return "FA3 kernel cached, liger-kernel importable: next, pod/speedcheck.py"


CHECKS = [("GPU", check_gpu, True), ("base model", check_model, True), ("data", check_data, True),
          ("no secrets", check_no_secrets, True), ("vLLM", check_vllm, True),
          ("fast path (optional)", check_fast_path, False)]


def main():
    failed = 0
    for name, check, required in CHECKS:
        try:
            print(f"ok    {name}: {check()}")
        except Exception as e:  # report every check, not just the first failure
            failed += required
            print(f"{'FAIL' if required else 'skip'}  {name}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
