# Training on a rented GPU pod

The Mac builds and checks the data; the pod only trains and samples. The pattern follows
`~/cs156-ml/pipeline1/pod/`. Two short sessions, with my ranking on the Mac in between:

| Session | On the pod | Time | Cost (1x H100, ~$2-4/h) |
|---|---|---|---|
| 1 | SFT A + B, voice-score and grade-model checks, sample answers (vLLM), score, regurgitation test | ~1 h with setup | ~$2-4 |
| (Mac) | I rank the answers at http://127.0.0.1:8765 | my time | $0 |
| 2 | DPO A + B on my picks | ~20-30 min | ~$1-2 |

## What goes where

| On the pod | Never on the pod |
|---|---|
| the code, cloned from GitHub (`ai_lab`, branch `main`); the masked SFT data, sent from the Mac by SSH after `privacy_check.py` passes; the base model (public) | raw CS156 data, the Forum cookie, any `.env`, unmasked text |
| while running: adapters, samples, the log | anything after `wipe.sh` |

## Environments

There are two environments, because vLLM 0.32 pins torch 2.13 and transformers below 5.20,
while training uses torch 2.14 and transformers 5.21.

- `~/minerva-gpt/.venv` is the training environment from `uv.lock`, exactly what the Mac ran.
- `~/vllm-env` is vLLM only. `generate.py --engine vllm` calls it with token ids, so it samples
  from exactly the prompts the training tokenizer made.

## Steps

**0. Rent the pod** (Prime Intellect, as for CS156; ~$2.36/h for an H100 in Oct 2026)

| Setting | Choose | Why |
|---|---|---|
| GPU | **1x H100 80GB**; an A100 80GB is fine if cheaper | 1B trains in minutes; the same pod fits the 7B later |
| Provider tier | a **datacenter / secure** provider, not a community or peer host | the masked class data sits on its disk while it runs |
| CPU | **x86-64** (not the GH200's Arm) | the torch and vLLM CUDA wheels are x86 first |
| Image | **Ubuntu 22.04 or 24.04 with CUDA 13** (NVIDIA driver >= 580) | uv.lock's torch 2.14 is built for CUDA 13; `setup.sh` stops on an older driver |
| Disk | **100 GB** | two environments, the model (6 GB), adapters |
| SSH key | your Mac's public key (`cat ~/.ssh/id_ed25519.pub`) | how push and pull reach it |

The dashboard then shows the SSH command, e.g. `ssh root@203.0.113.7 -p 22`. Below, that
address is `HOST` and the port is `PORT`.

**1. Mac: code from GitHub, data from the Mac** (~1 min)
```
cd ~/ai_lab/experiments/minerva-gpt
bash pod/push.sh HOST PORT
```
It refuses if the privacy gate fails, and if this folder differs from GitHub `main` (push and
merge first, so the pod runs exactly what you see). `CODE=mac` sends this Mac's files instead.

**2. Pod: set up and check** (~10 min)
```
ssh HOST -p PORT
bash ~/minerva-gpt/pod/setup.sh
cd ~/minerva-gpt && uv run python pod/preflight.py      # must print "all passed"
GEN_ENGINE=vllm sh smoke_test.sh                          # whole pipeline on fake data, this GPU
uv run python pod/speedcheck.py                           # turns the fast path on only if proven (~5 min)
```

**3. Pod: session 1** (~30-40 min)
```
tmux new -s minerva
bash ~/minerva-gpt/pod/run.sh
```
- Detach with `Ctrl-b` then `d`. Come back with `tmux attach -t minerva`.
- Note the two "suggested --voice-gate" lines and the AUC verdicts.

**4. Mac: bring results back, then wipe and terminate the pod**
```
bash pod/pull.sh HOST PORT
ssh HOST -p PORT 'bash ~/minerva-gpt/pod/wipe.sh'        # then terminate it in the dashboard
```

**5. Mac: rank**
```
uv run python rank_server.py --a data/generations/A-val.jsonl --b data/generations/B-val.jsonl
```
Open http://127.0.0.1:8765. Rank as many as you like, then try the blind test.

**6. Session 2: DPO.** Rent a pod again, then steps 1-2 (push now also sends the rankings, the
adapters and the samples), then:
```
GATE_A=<from session 1> GATE_B=<from session 1> bash ~/minerva-gpt/pod/run_dpo.sh
```
Leave a variant's gate unset if its voice-score AUC check failed. Then step 4 again.

## Speedups

| Speedup | How | On when |
|---|---|---|
| vLLM sampling | own environment, same token ids as the training tokenizer | always |
| bf16, bigger batches, no gradient checkpointing | `run.sh` flags | always (speed only) |
| **FlashAttention 3** | prebuilt Hopper kernel from the kernels hub (`kernels-community/vllm-flash-attn3`); no compiling | after `speedcheck.py` passes |
| **Packing** (no padding) | TRL padding-free; FlashAttention's variable-length mode keeps packed examples apart | after `speedcheck.py` passes |
| **Liger kernels** | fused lm_head + loss: the 100K-word logits are never materialized | after `speedcheck.py` passes |

`speedcheck.py` proves two things on this pod before `run.sh` uses the fast path:
1. **Isolation.** Each example gives the same log-probs packed behind another as alone. A
   negative control, the same row without position restarts, must differ. On the Mac, sdpa
   fails this check (2.50 per token), as it should.
2. **Equivalence.** One epoch on the same 640 examples, verified path vs fast path: val loss
   within 2%, and actually faster.

If either fails, `run.sh` stays on the verified sdpa path. `BATCH=16 bash pod/run.sh` trades
memory for speed.
