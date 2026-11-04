# Next session: the context test is done, build on `full` (handoff, 4 Nov 2026)

Start a new chat with: *"Read `experiments/minerva-gpt/NEXT_SESSION.md` and `LIVE_CLASS_PLAN.md`, then
help me with the next step."* Everything needed is here or linked.

## Decision (4 Nov 2026): use `full`, no retraining

**The live bot builds its prompts as `full`: my intro (tag, class header, activity, LOs/HCs, readings,
pre-class work, poll on screen), then `Class so far:` with everything said up to that moment, then the
ask and replies.** The weights don't change. This is about what goes in the prompt.

The test ran on an RTX Pro 6000 on 3-4 Nov: 618 held-out spoken answers from 84 classes. vLLM matched
plain Transformers on long prompts (average gap 0.0025, and 0.0022 over 8,192 tokens; the limit was
0.01), so the numbers can be trusted.

| | `window` (control) | `tail` | **`full`** | `full-top` | `full-tag` | `full-pcw` | Me |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Verdict | | PASS | **UNSURE** | FAIL | FAIL | UNSURE | |
| Loss on my answer | 2.609 | 2.528 | **2.514** | 2.525 | 2.537 | 2.514 | |
| vs window | | -0.081 | **-0.095** | -0.084 | -0.072 | -0.095 | |
| vs window, over 8,192 tokens | | -0.074 | **-0.092** | -0.081 | -0.065 | | |
| Length vs window | | 1.02x | 1.12x | 0.90x | 1.21x | 1.07x | |
| Median words | 61 | 65 | 75 | 53 | 75 | 71 | 55 |
| "like" per 1,000 | 58.9 | 60.5 | 60.0 | 63.6 | 58.3 | 60.1 | 44.7 |
| "I think" per 1,000 | 10.6 | 11.1 | 10.9 | 12.7 | 10.6 | 11.0 | 5.9 |
| Cut off at the length limit | 7% | 8% | 9% | 5% | 11% | 8% | |
| Answers stuck in a loop | 81 | 60 | 55 | 44 | 64 | 54 | |
| Tried to write another speaker | 0 | 0 | 0 | 0 | 0 | 0 | |

`full-tag` is `full` with `Answer as: Brady` repeated right before `<|assistant|>`. `full-pcw` is `full`
with all of my pre-class work and each question in full, instead of the first 300 words and 12 words per
question; 299 prompts changed.

**What was found:**
- **The whole class helps.** `full` has the best loss at every prompt length. The gain is about three
  times the gap between models A and B (0.029), and it holds over 8,192 tokens.
- **The voice tag being out of reach didn't hurt.** `full` leaves it out of reach of 24 of the 32 layers
  on 76% of prompts, with habits no further from mine than the window. Keeping it in reach (`full-top`)
  or repeating it (`full-tag`) was worse on loss, so don't add either without retesting.
- **Structure matters.** The same text in a different order, or one extra line, moved loss and length,
  so test any new prompt format first.
- **Pre-class work doesn't appear to need all 32 layers.** On prompts over 4,096 tokens, `full` minus
  `full-top` is -0.013 with pre-class work and -0.012 without (not randomly assigned, so strong support
  rather than proof).
- **Whole pre-class work made no measurable difference:** -0.0012 on loss vs `full` where the text changed
  (95% -0.0028 to +0.0001), and answer length within noise. It costs a median of 239 tokens, so it's
  optional. This doesn't show what pre-class work adds in itself: no version removed it.

**Accepted or still open:**
- **`full` is UNSURE on length only.** Answers run longer than mine (median 75 words vs 55), mostly because
  it gives too few short answers: its longest 10% start at 184 words vs my 180. A stop boost on the
  end-of-answer token (`logit_bias`) was considered and declined.
- **"like" and "I think" are still too high** (about 1.3x and 1.8x my rate on the window too). That comes
  from training, and adding class context doesn't fix it.
- **Blind picks are too few to count:** 4 screens so far (`full` 2, `window` 2, `tail` 0), and about 20 are
  needed. Rerun the page on the Mac (no pod needed):
  `uv run python rank_server.py --model window=data/generations/7b/B-val-ctx-window.jsonl --model tail=data/generations/7b/B-val-ctx-tail.jsonl --model full=data/generations/7b/B-val-ctx-full.jsonl --source class_turn --per-model 1 --data data/judge-context`,
  plus `--exclude` for each earlier `data/judge-7b/round*/rankings.jsonl` and for
  `data/generations/7b/quick/context-test-answers.jsonl`.
- **Not tested:** whether retraining would do better, and how this behaves live, since every prompt came
  from a moment when I actually spoke. First-word delay on prompts this long is also unmeasured.

**Measured speed** (RTX Pro 6000, FlashAttention 2, the existing settings): about 21,000 tokens/s scoring.
The four-version run took 21 minutes. Triton was 24% slower. Bigger batches and turning off decoding gave no
gain, and FlashInfer can't run this model (mixed sliding and full layers). `full-tag` and `full-pcw` ran
from side scripts, kept with the logs in `runs/context-test-3nov/` (gitignored), not in the repo.

## Where things stand

- **The model.** OLMo 3 7B base, LoRA SFT then SPIN DPO on my own class data. The chosen model is
  **variant B** (my answers x3 + anonymized, tagged classmates x1) **with DPO**. It won on how I
  speak (word mix, length) and on my blind picks; B beat A on 65-68% of my held-out spoken answers.
- **Where it lives (all private on Hugging Face):** `bradyrutherford/minerva-olmo3-7b-B` (`main` =
  merged SFT + DPO, bf16, 14.6 GB, plus `adapters/sft` and `adapters/dpo`; branch `sft` = SFT only),
  the same for variant A, and `bradyrutherford/minerva-olmo3-7b-B-mlx-8bit` for the Mac (`chat.py`).
- **The full write-up** (data, LoRA on all 7 linear layers of all 32 blocks, DPO, results, decision
  log): a private Claude doc, https://claude.ai/artifact/CXBk6qpDn7mbfetS9bqdzD
- **Merged:** PR #5 (H200 throughput, chat.py, 8-bit copy, rank page), PR #6 and PR #7 (the context test).

## The context test: what it was (for the record)

The live bot (`LIVE_CLASS_PLAN.md`) wants the class so far in its prompt. The model was trained on
short windows (median 743 tokens) and its replies were only ever my words, then end-of-text. The
question: **does it answer as well, and still sound like me, with the class in front of it, without
retraining?** This tests the current model; nothing is trained.

`eval_context.py` scores the same model on my **618 held-out spoken answers** (from the 84 validation
classes with an answer of mine) under four prompts. Sizes measured on the real prompts:

| Variant | Prompt | Median tokens | Prompt + answer within 4,096 | Voice tag in reach of all 32 layers |
| --- | --- | --- | --- | --- |
| window (control) | Exactly as trained | 743 | 100% | 100% |
| tail | The last ~2,500 tokens of the class, before the ask | 3,111 | 100% | 100% |
| full | The whole class so far, before the ask | 7,410 (longest ~19,000) | 23% | 23% |
| full-top | The whole class so far, at the very top; the trained prompt after it, untouched | 7,410 | 23% | 100% |

**Why placement matters.** 24 of OLMo 3 7B's 32 layers see only the last 4,096 tokens; the other 8 see
everything (`sliding_window` and `layer_types` in its config). The first line of every prompt is the
voice tag, `Answer as: Brady`. So `full` puts the tag and my PCW out of those layers' reach on 77% of
answers. If `full` does badly, `tail` and `full-top` say whether length or placement is to blame, which
could save a retraining.

The class so far (`data/context/val.jsonl`, built by `prepare_data.py`) is everything said before the
professor's ask, masked, as `Who: text` lines, in a `Class so far:` block. Every prompt ends exactly as
in training. Checked on the real data: the answer being scored never appears in its prompt (0 of 618).

**Evals:**
1. **Loss on my real answer (main result):** same answer tokens under each prompt, paired against the
   window, 95% interval by resampling whole classes, overall and by prompt length (up to 4,096: 141
   answers; 4,097-8,192: 195; over 8,192: 282). Every answer's loss is saved, not only the averages.
2. **One generated answer per prompt** (temperature 0.8, top-p 0.95, up to 256 tokens): my habits per
   1,000 words, length vs mine, cut-offs, loops, overlap with my real answer (content words, meaning).
3. **Tried to write another speaker:** generation stops where an answer would start a new
   `Professor:` / `Student N:` / `Speaker:` / `Brady:` line (the same guard the live bot uses), and
   the test counts how often each variant tried.
4. **My blind picks,** `window` vs the best variant on the rank page (`B-val-ctx-<variant>.jsonl`).
   Optional if a variant passes; this settles an UNSURE.

**Decision, fixed before seeing results.** The test prints PASS, FAIL or UNSURE per variant against the
window. The limits come from earlier runs, so that chance differences can't decide it:

| Measure | PASS | UNSURE | FAIL | Why this limit |
| --- | --- | --- | --- | --- |
| Loss vs window | All of the 95% interval below +0.015 | Interval crosses +0.015 | All of it above | B vs A differ by 0.029, so half of that |
| My habits (further from my rate, interval above 0) | None | One | Two or more | Two runs of one model differ by up to about 5 "like"s per 1,000 |
| Length vs window | Interval inside about 10% | Interval crosses it | All of it outside | Two runs of one model differ by about 6% |
| Other speakers vs window | Interval below +2% of answers | Interval crosses it | All of it above | 0 of 1,236 answers on the trained prompt |

The live bot uses the variant with the most class that passes, and the length table has to agree: a
collapse over 8,192 tokens rules out `full` and `full-top` even if the average is fine.

| tail | full | full-top | Meaning | Next |
| --- | --- | --- | --- | --- |
| Pass | Pass | Pass | The model handles the whole class | No retraining; `LIVE_CLASS_PLAN.md` phase 1 |
| Pass | Fail | Pass | Placement was the problem | Use `full-top`; no retraining |
| Pass | Fail | Fail | Length is the problem | Use the 2,500-token tail; retrain only if the bot needs the whole class |
| Fail | any | any | Any added class hurts | The full-class retraining below |

## The pod

- **GPU:** one **NVIDIA RTX Pro 6000, 96 GB** ($2.73/hour; the run itself took 21 minutes, and the pod
  was up about 1.5 hours with all benchmarks and extra versions, about $4.20). Any 80 GB H100 or A100 also works. Needs driver 580+ (CUDA
  13); `setup.sh` checks. It is a Blackwell card (SM120): by vLLM's code it defaults to FlashAttention
  2. If that won't run, `run_context_test.sh` retries with Triton attention, through `attention_config`
  because vLLM 0.32 ignores the old `VLLM_ATTENTION_BACKEND` variable.
- **Commands, from the Mac** (`HOST` and `PORT` from the provider; images log in as `ubuntu` or `root`):

```
cd ~/ai_lab/experiments/minerva-gpt
CONTEXT_ONLY=1 bash pod/push.sh HOST PORT            # privacy gate (~6 min), code from GitHub main, validation data only (22 MB)
ssh -p PORT HOST 'bash ~/minerva-gpt/pod/setup.sh'   # once on a fresh pod: driver check, both environments
# The quick pass: download the model, checksum, vLLM vs plain Transformers, 30 answers end to end
ssh -p PORT HOST 'cd ~/minerva-gpt && HF_TOKEN=$(cat) LIMIT=30 bash pod/run_context_test.sh' < ~/.cache/huggingface/token
# The full run, in the background so a sleeping Mac can't kill it; no login needed any more
ssh -p PORT HOST 'setsid nohup bash ~/minerva-gpt/pod/run_context_test.sh </dev/null >>~/context-test.log 2>&1 &'
ssh -p PORT HOST 'tail -f ~/context-test.log'        # ends with "context test done" or "context test FAILED"; Ctrl-C only stops watching
bash pod/pull.sh HOST PORT                           # results -> data/generations/7b/; compares checksums and says if it's not safe to wipe
ssh -p PORT HOST 'bash ~/minerva-gpt/pod/wipe.sh'    # only after pull.sh says done; then terminate the pod in the dashboard
```

`HOST` is `user@address` (for example `ubuntu@1.2.3.4`). `pod/README.md` has the general walkthrough.

**What protects the money and the result:**
- **The crosscheck runs first.** vLLM has no OLMo 3 code of its own: it runs through its Transformers
  wrapper, and this path has never run on prompts this long. So it scores 30 prompts both ways and
  stops with exit 3 if they differ by more than 0.01 on average (for scale, the 8-bit copy and the
  full model differ by 0.003), or by more than 0.005 in one direction. That holds separately for all
  prompts, those over 4,096 tokens and those over 8,192. It also compares greedy answers on long
  prompts, with vLLM's sampling settings and from its prefix cache. A control with the 4,096 limit
  off shows how far off a vLLM that ignored the limit would be.
- **A second check is free:** the window's losses are compared answer by answer with the plain
  Transformers run saved in `runs/perrow-7b.jsonl` (spoken answers averaged 2.609).
- **Longest prompts are scored first,** so running out of memory shows in minutes.
- **A crash loses little:** scores are saved after every 64 answers and the main result before
  generation starts. Rerun the same command and it picks up where it stopped.

**Before the pod:** `uv run python smoke_context.py` tests the logic and every failure path with no GPU
and no class data, against a stand-in for vLLM (`pod/fake_vllm`). It can't prove vLLM itself: only
the crosscheck can, and the Blackwell attention backend also stays untested until then.

Results: `context-test.json` (a table is also printed), `context-test-answers.jsonl`,
`context-crosscheck.json`, `context-backend.txt` and the four `B-val-ctx-*.jsonl` files. The quick
pass writes to `quick/` beside them, so it can't mix with the full run.

**Earlier generations shared a seed.** Until this test, `vllm_sample.py` gave every prompt `seed=0`, so
all answers got the same noise at each word position. Each answer was still a fair sample, but
averages over answers (habit rates, length) were noisier than they looked, and the SPIN pairs and blind
picks came from those runs. The MLX temperature test doesn't use this code and also found "like" at
about 1.5x my rate, so that stands. Each prompt now gets its own seed.

## If a retrain is ever wanted: the full-class model

The test did not call for this. Kept as the plan in case the live class shows otherwise.

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
