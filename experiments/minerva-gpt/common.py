"""Shared settings for the pipeline: paths, prompt tags, chat template, training defaults.

Standard library only, so prepare_data.py (run in the CS156 environment) can import it too.
"""

import json
import os
from pathlib import Path

# Pipeline scripts never touch the network: a missing file must fail loudly instead of
# starting a silent multi-GB download mid-run, and nothing about the data leaves the Mac.
# Fetch models once with download_model.py, which turns this off for itself.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).parent
DATA = ROOT / "data"                # gitignored: everything derived from class recordings
CHECKPOINTS = ROOT / "checkpoints"  # gitignored: adapters can regurgitate training text

BASE_MODEL = "allenai/OLMo-2-0425-1B"
ME = "Brady"
TAG_ME = "Answer as: Brady"
TAG_CLASSMATE = "Answer as: a classmate"

# OLMo 2's base tokenizer ships without a chat template. This is the one OLMo 2's own
# instruct models use (allenai/OLMo-2-0425-1B-SFT); its role markers are plain text.
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{% if not loop.last %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
    "{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}"
    "{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"
)

# LoRA on every linear layer, MLP included, at about 10x the full fine-tuning learning rate
# (Thinking Machines, "LoRA Without Regret", 2025). Variant B sees my examples 3x and each
# classmate example once; A sees mine 3x, so both get the same exposure to my data.
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": "all-linear"}
LEARNING_RATE = 2e-4
EFFECTIVE_BATCH = 32
MAX_LENGTH = 2048
MY_REPEATS = 3
CLASSMATE_REPEATS = 1
GRADE_BONUS_Z = 0.5  # my examples graded this far above that grader's mean get one extra copy


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sft_rows(variant, split, data=DATA):
    """The SFT mixture for one variant: A = my examples only, B = mine plus classmates'."""
    mine = read_jsonl(data / f"sft/{split}.jsonl")
    rows = []
    for row in mine:
        bonus = int(row.get("grade_z") is not None and row["grade_z"] >= GRADE_BONUS_Z)
        rows += [row] * (MY_REPEATS + bonus if split == "train" else 1)
    if variant == "B":
        classmates = read_jsonl(data / f"sft_classmates/{split}.jsonl")
        rows += classmates * (CLASSMATE_REPEATS if split == "train" else 1)
    return rows


def load_tokenizer(name=BASE_MODEL):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if not tok.chat_template:
        tok.chat_template = CHAT_TEMPLATE
    return tok


def device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(*adapters, merge_last=False, name=BASE_MODEL, attn=None):
    """The base model in bfloat16 on the best device, with LoRA adapters applied in order.
    Every adapter but the last is merged into the weights (SFT under DPO); the last stays a
    separate adapter unless `merge_last`, so it can be switched off to get the model below it."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {"attn_implementation": attn} if attn else {}
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, **kwargs).to(device())
    if adapters:
        from peft import PeftModel

        for i, adapter in enumerate(adapters):
            model = PeftModel.from_pretrained(model, str(adapter))
            if i < len(adapters) - 1 or merge_last:
                model = model.merge_and_unload()
    return model


def lora_config():
    from peft import LoraConfig

    return LoraConfig(task_type="CAUSAL_LM", **LORA)
