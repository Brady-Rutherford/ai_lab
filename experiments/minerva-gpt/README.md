# Minerva-GPT

A model that answers like me in class and in assignments. It takes an open base model (OLMo)
and post-trains it on data only I have: three years of Minerva class transcripts, my
pre-class work and poll answers, and my professors' grades on what I said and wrote.

Following Karpathy's "Deep Dive into LLMs" structure: pretraining gives knowledge and fluency
(here: OLMo's, from trillions of tokens), and post-training gives the persona (here: me).
Pretraining our own base with nanochat comes later, as a comparison.

## Privacy rules (hard)

**This repo is public. Nothing derived from classes is ever committed or uploaded.**

- Transcripts, SFT data, grades, rankings, generations and trained adapters live under
  `data/` and `checkpoints/`, both gitignored. They stay on this Mac, or go straight to a
  rented GPU box by rsync.
- **Masking.** Names are masked by the CS156 roster masker, then by a second pass over every
  roster (`names.py`). `privacy_check.py` must find 0 definite hits before any data is used.
- **Commit guard.** A local pre-commit hook (`.git/hooks/pre-commit`, not committed) refuses
  data and weight files and `.env` files, and scans staged text for roster names.
- **No network, no logging.** Pipeline scripts run offline (`HF_HUB_OFFLINE`), wandb is off,
  and the ranking page binds to 127.0.0.1 and loads nothing external.
- **Never paste cookies or tokens into a chat.** The Forum cookie lives in `~/.minerva/.env`.

## The pipeline

```
CS156 archive ─► prepare_data.py ─► privacy_check.py ─► train_sft.py ─► generate.py
 (transcripts,     ask → my answer      0 definite         LoRA on OLMo     n answers per
  PCW, polls,      + the class's        name hits          2 1B base        held-out ask
  grades, LOs)     LOs/HCs + grade                                         (+ retrieve.py)
                                                                                │
   train_dpo.py ◄── rank_server.py ◄── voice_score.py + reward_model.py ◄───────┘
   (DPO on top       (I judge, blind:   (log-prob voice contrast;
    of SFT)          voice yes/no,       grade model on content,
                     then best)          hashtags stripped)
```

Two adapters on one base:

| Variant | SFT data | Leaves the Mac? |
|---|---|---|
| **A** | my answers only, seen 3x | could be shown to others |
| **B** | my answers 3x + classmates' class answers 1x, tagged "Answer as: a classmate" (one pooled tag, personal disclosures dropped) | never |

**Preference stage: your voice decides, grades refine, retrieval informs.**

- **Voice.** Whatever its grade, an answer I mark "not my voice" is never chosen.
- **Grades.** They rank only among answers that sound like me.
- **Retrieval.** My own best-graded past answers on the same LOs/HCs go into the prompt.
- **Voice score.** `log p(answer | "Answer as: Brady") − log p(answer | "Answer as: a classmate")`.
  It is trusted only if it separates my real held-out answers from classmates' (AUC ≥ 0.8).

## Setup

```bash
cd experiments/minerva-gpt
uv sync                                     # training env: torch (MPS), transformers, peft, trl
uv run python download_model.py             # OLMo 2 1B base into the HF cache, size-verified
```

Data prep and the privacy check run inside the CS156 environment, because they reuse its
`oncue` masker and Forum clock: `uv run --project ~/cs156-ml python …`.

## Running it, step by step

```bash
# 1. data (A and B are built together), then the privacy gate
uv run --project ~/cs156-ml python prepare_data.py
uv run --project ~/cs156-ml python privacy_check.py data/sft data/sft_classmates

# 2. SFT: a debug run first, then both variants
uv run python train_sft.py --variant A --limit 500
uv run python train_sft.py --variant A
uv run python train_sft.py --variant B

# 3. evals (validate prints the AUC and a suggested --voice-gate for step 4)
uv run python voice_score.py --adapter checkpoints/sft-B --variant B --validate val
uv run python reward_model.py --train
uv run python generate.py --adapter checkpoints/sft-A --variant A --n 2 --retrieve
uv run python generate.py --adapter checkpoints/sft-B --variant B --n 2 --retrieve
uv run python eval_regurgitation.py data/generations/A-val.jsonl data/generations/B-val.jsonl

# 4. I rank, blind (http://127.0.0.1:8765), then DPO
uv run python rank_server.py --a data/generations/A-val.jsonl --b data/generations/B-val.jsonl
uv run python train_dpo.py --variant B --rankings data/rankings.jsonl \
    --auto data/generations/B-val.voice.grade.jsonl --voice-gate <from step 3>
```

`sh smoke_test.sh` runs every step above on fake data (`synthetic.py`) in a temp folder; it
reads no class data and writes nothing into `data/` or `checkpoints/`.

**Speed on the M4:**
- LoRA SFT runs at about 240 tokens/s with gradient checkpointing (measured on short fake
  examples).
- At that rate, a full variant A run is roughly 5 hours and B roughly 14. The 500-example
  debug run measures it on real lengths.
- One rented GPU would take minutes.

## On the GPU (later, OLMo 3 7B)

Before each compute step, check for an existing optimization that matches this exact setup
without changing results:

- **vLLM** for sampling (`generate.py`), with LoRA adapter serving;
- **flash-attention 2/3**, and with it TRL's padding-free packing. These are off on the Mac,
  where SDPA attention would let packed examples see each other;
- **Liger kernels** for SFT and DPO memory;
- TRL's own vLLM integration, if we add an RL stage.

Move data by rsync over SSH, use no persistent volumes, and terminate the box after each run.

## Waiting on

- **Assignment harvest:** instructions plus my submissions, split by section. It needs a fresh
  Forum cookie in `~/.minerva/.env` and endpoint discovery, done together.
- **HC handbook:** put it in `data/handbook/` (gitignored), to add good-use examples to the
  LO/HC context.
- **Pretraining track:** nanochat (cloned in `nanochat/`, pinned at `92d63d4`, gitignored).
  Crawl → filtering → tokenizer study → our own base → the same post-training → compare.
