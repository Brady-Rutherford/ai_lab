# Next session: run the context test (handoff, 3 Nov 2026)

Start a new chat with: *"Read `experiments/minerva-gpt/NEXT_SESSION.md`, then run the context test on
the pod I give you."* Everything needed is here or linked.

## Where things stand

- **The model.** OLMo 3 7B base, LoRA SFT then SPIN DPO on my own class data. The chosen model is
  **variant B** (my answers x3 + anonymized, tagged classmates x1) **with DPO**. It won on how I
  speak (word mix, length) and on my blind picks; B beat A on 65-68% of my held-out spoken answers.
- **Where it lives (all private on Hugging Face):** `bradyrutherford/minerva-olmo3-7b-B` (`main` =
  merged SFT + DPO, bf16, 14.6 GB, plus `adapters/sft` and `adapters/dpo`; branch `sft` = SFT only),
  the same for variant A, and `bradyrutherford/minerva-olmo3-7b-B-mlx-8bit` for the Mac (`chat.py`).
- **The full write-up** (data, LoRA on all 7 linear layers of all 32 blocks, DPO, results, decision
  log): a private Claude doc, https://claude.ai/artifact/CXBk6qpDn7mbfetS9bqdzD
- **Merged:** PR #5 (H200 throughput, chat.py, 8-bit copy, rank page) and PR #6 (this test).

## The context test: what and why

The live bot (`LIVE_CLASS_PLAN.md`) wants the class so far in its prompt. The model was trained on
short windows (median 743 tokens) and its replies were only ever my words, then end-of-text. The
question: **does it answer as well with the class in front of it, without retraining?**

`eval_context.py` scores the same model on my **618 held-out spoken answers** (93 validation
classes) under three prompts:

| Variant | Prompt | Size |
| --- | --- | --- |
| window (control) | Exactly as trained | ~900 tokens |
| tail | Window + the last ~4,000 tokens of the class so far | ~1,100-4,900 |
| full | Window + the whole class so far (capped at 28,000 tokens) | up to ~21,000 |

The class so far (`data/context/val.jsonl`, built by `prepare_data.py`) is everything said before the
professor's ask, masked, as `Who: text` lines. It goes in as a `Class so far:` block just before
`Professor: <ask>`, so every prompt ends exactly as in training. Checked on the real data: the
answer being scored never appears in it (0 of 618).

**Evals:**
1. **Loss on my real answer (main result):** same answer tokens under each prompt, paired against
   the window, 95% interval by resampling whole classes, plus the share of answers where it wins.
2. **One generated answer per prompt** (temperature 0.8, top-p 0.95, up to 256 tokens): my habits per
   1,000 words, length vs mine, cut-offs, loops, overlap with my real answer (content words, meaning).
3. **Tried to write another speaker:** generation stops where an answer would start a new
   `Professor:` / `Student N:` / `Speaker:` / `Brady:` line (the same guard the live bot uses), and
   the test counts how often each variant tried.
4. Optional: my blind picks, `window` vs `full` on the rank page (`B-val-ctx-<variant>.jsonl`).

**Decision:** if `tail` or `full` is no worse on loss and does not try to write other speakers much
more than `window`, the live bot uses it as is (go to `LIVE_CLASS_PLAN.md`, phase 1). If worse, do
the full-class retraining below.

## The pod

- **GPU:** one **NVIDIA RTX Pro 6000, 96 GB** ($2.73/hour; about an hour, so about $3). Any 80 GB
  H100 or A100 also works. Needs driver 580+ (CUDA 13); `setup.sh` checks. It is a Blackwell card:
  if vLLM's default attention fails, `run_context_test.sh` retries with `VLLM_ATTENTION_BACKEND=TRITON_ATTN`.
- **Commands, from the Mac** (`HOST` and `PORT` from the provider; images log in as `ubuntu` or `root`):

```
cd ~/ai_lab/experiments/minerva-gpt
bash pod/push.sh HOST PORT                           # privacy gate, code from GitHub main, masked data incl. data/context
ssh -p PORT HOST 'bash ~/minerva-gpt/pod/setup.sh'   # once on a fresh pod: driver check, both environments
ssh -p PORT HOST 'cd ~/minerva-gpt && HF_TOKEN=$(cat) bash pod/run_context_test.sh' < ~/.cache/huggingface/token
bash pod/pull.sh HOST PORT                           # results -> data/generations/7b/context-test.json
ssh -p PORT HOST 'bash ~/minerva-gpt/pod/wipe.sh'    # then terminate the pod in the dashboard
```

`HOST` is `user@address` (for example `ubuntu@1.2.3.4`). `pod/README.md` has the full walkthrough.

`run_context_test.sh` downloads the merged model from the private repo (the HF login is piped in,
never written on the pod), checks its SHA-256, then runs scoring and generation. `LIMIT=100` makes a
quick first pass; `TAIL=` changes the middle variant. Results: `context-test.json` (a table is also
printed) and the three `B-val-ctx-*.jsonl` files.

Untested parts, fix on the pod if they break: vLLM prompt log-probs on very long prompts (scoring
runs in chunks of 64 with smaller prefill chunks) and the Blackwell attention backend. Everything
else was dry-run on the Mac with a stand-in for vLLM.

## If the test says retrain: the full-class model

- **Format:** one sequence per class, the whole masked transcript with speaker tags (`Professor:`,
  `Student N:`, `Brady:`), class header, readings and my PCW at the top. The loss covers my lines
  (weight 1) and classmates' lines (weight 1/3, B's ratio); the professor's lines are context only.
  Every one of my turns counts, not only answers to a question.
- **Turn end:** an explicit end-of-turn marker trained after each of my lines; generation starts after
  `Brady:` and stops at that marker, with the same other-speaker stop rule as a backup.
- **Training:** the same LoRA (all-linear, rank 16, alpha 32, LR 1e-4), then SPIN DPO with the settings
  that won (length-normalized loss + SFT term, beta 0.1, LR 2.5e-5; vLLM prefix caching for sampling).
  FlashAttention 2, Liger fused loss, gradient checkpointing. About 6-8 GPU hours on an 80-96 GB card.
- **Split:** 100 test classes drawn from the old held-out classes (shared with On Cue's rerun), ~25
  validation classes for checkpoints and gates, everything else train; no k-fold for the LLM.
- **Data:** includes the new classes below once ingested.

## Other open threads

- **On Cue (cs156-ml):** 16 new classes synced from Forum on 3 Nov; 7 usable (5 CS156, 2 IL18102)
  appended to `split.csv` as train on branch `pipeline1-new-classes` (existing rows unchanged). Their
  ingest (Deepgram transcripts, clock sync, hand-raise scan, QA) was running at handoff; then commit
  that branch with a PR and rebuild this project's data with `prepare_data.py` (the new classes are
  train, so the context test's validation data does not change). Rerun plan: `pipeline1/RERUN_PLAN.md`
  (cs156-ml PR #7).
- **Sampling temperature:** `chat.py` defaults to 0.8; the test on 40 questions favoured 1.0 (vocabulary,
  fewer loops) but did not fix the extra "like" (about 1.5x mine at every temperature).
- **The test split** (1,025 of my answers) is still untouched: score it once at the end.

## Rules (unchanged)

- ai_lab is a **public** repo: class data, transcripts, checkpoints and generations never go in it
  (gitignored; the pre-commit hook checks names). Masked data goes only to the Mac and the rented GPU.
- Weights only on **private** Hugging Face repos. Pipe the HF login into the one command that needs
  it; never write it on the pod. Never use mlx_lm's `--upload-repo` (it creates public repos).
- The Forum cookie lives in `~/.minerva/.env` (`~/.minerva/set_secret` saves it from the clipboard).
- wandb off. Wipe the pod and terminate it after every run.
