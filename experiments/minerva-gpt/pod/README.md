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
| this folder's code, the masked SFT data (after `privacy_check.py` passes), the base model (public) | raw CS156 data, the Forum cookie, any `.env`, unmasked text |
| while running: adapters, samples, the log | anything after `wipe.sh` |

## Environments

There are two environments, because vLLM 0.32 pins torch 2.13 and transformers below 5.20,
while training uses torch 2.14 and transformers 5.21.

- `~/minerva-gpt/.venv` is the training environment from `uv.lock`, exactly what the Mac ran.
- `~/vllm-env` is vLLM only. `generate.py --engine vllm` calls it with token ids, so it samples
  from exactly the prompts the training tokenizer made.

## Steps

**0. Rent the pod.**
- 1x H100 80GB (or A100 80GB), x86-64, an Ubuntu image with CUDA, 100 GB disk, your SSH key.
- The dashboard shows the SSH command, e.g. `ssh root@203.0.113.7 -p 22`. Below, that address
  is `HOST` and the port is `PORT`.

**1. Mac: send the code and the data** (~1 min; refuses unless the privacy gate passes)
```
cd ~/ai_lab/experiments/minerva-gpt
bash pod/push.sh HOST PORT
```

**2. Pod: set up and check** (~10 min)
```
ssh HOST -p PORT
bash ~/minerva-gpt/pod/setup.sh
cd ~/minerva-gpt && uv run python pod/preflight.py      # must print "all passed"
GEN_ENGINE=vllm sh smoke_test.sh                          # whole pipeline on fake data, this GPU
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

## Optional speedups, verified before use

- **FlashAttention 2 + packing** (`train_sft.py --attn flash_attention_2 --packing`). Install
  flash-attn into `.venv`. Before switching, run 20 steps both ways and check that the loss
  curves match. With plain sdpa, packed examples would attend to each other, so `--packing`
  refuses without FlashAttention.
- **Bigger batches**: `BATCH=16 bash pod/run.sh` if memory allows.
