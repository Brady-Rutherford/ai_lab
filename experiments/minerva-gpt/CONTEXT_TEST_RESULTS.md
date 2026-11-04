# Context test: all findings and comparisons (3-4 Nov 2026)

Does the trained model (OLMo 3 7B, variant B, SFT + DPO) answer as well, and still sound like me, with the class so far in its prompt? The decision and what it means for the bot are in [NEXT_SESSION.md](NEXT_SESSION.md). This file holds every number behind it. Aggregate numbers only: no class text, answers or prompts, which stay on the Mac.

**Data:** 618 held-out spoken answers from 84 validation classes. Nothing was trained. **Hardware:** NVIDIA RTX Pro 6000 Blackwell, 96 GB, vLLM 0.32 with FlashAttention 2. One sampled answer per prompt (temperature 0.8, top-p 0.95, up to 256 tokens), with one seed per answer, the same under each prompt.

## The versions

24 of the model's 32 layers see only the last 4,096 tokens; the other 8 see everything. "Intro" means the tag `Answer as: Brady`, class header, activity, LOs/HCs, readings (first 200 words), my pre-class work (first 300 words; each question cut to 12) and any poll on screen.

| Version | Order of the prompt | Median prompt tokens | Prompt + answer within 4,096 |
| --- | --- | --- | --- |
| `window` (control) | intro, ask + replies: exactly as trained | 743 | 100% |
| `tail` | intro, last 2,500 tokens of class, ask + replies | 3,111 | 100% |
| `full` | intro, whole class so far, ask + replies | 7,410 | 23% |
| `full-top` | whole class so far, intro, ask + replies | 7,410 | 23% |
| `full-tag` | `full`, plus `Answer as: Brady` again right before `<|assistant|>` | 7,415 | about the same as `full` |
| `full-pcw` | `full`, with all my pre-class work and each question in full | not recorded | about the same as `full` |

`full-tag` and `full-pcw` ran afterwards from side scripts (kept with the logs on the Mac, gitignored), paired with the first four by the same seeds. `full-pcw` ran only the 299 prompts whose text changes; the other 319 are the `full` prompt, so they take `full`'s result.

## Verdicts

The rule was fixed before seeing results: see NEXT_SESSION.md. Limits against the window: loss +0.015, length about 10%, other speakers +2% of answers, and two or more habits further from my rate is a FAIL.

| Version | Verdict | Why |
| --- | --- | --- |
| `tail` | PASS | passes every check |
| `full` | UNSURE | answer length may have moved more than about 10% |
| `full-top` | FAIL | further from my habits: like, I think, yeah / yep; answer length may have moved more than about 10% |
| `full-tag` | FAIL | answer length moved more than about 10% |
| `full-pcw` | UNSURE | answer length may have moved more than about 10% |

**Decision: `full`.** It has the best loss of the first five versions at every prompt length, and `full-pcw` matches it within noise. Its open issue, answer length, is accepted for now. A stop boost to shorten answers was considered and declined.

## Is vLLM right on long prompts?

The same prompts through vLLM and through plain Transformers, which uses neither vLLM's cache nor its chunking. Difference in loss:

| Prompts | Count | Average gap (limit 0.01) | Average signed gap (limit 0.005) | Largest on one answer |
| --- | --- | --- | --- | --- |
| all | 30 | 0.0025 | +0.0003 | 0.0108 |
| over 4,096 | 18 | 0.0023 | -0.0003 | 0.0069 |
| over 8,192 | 11 | 0.0022 | -0.0000 | 0.0068 |

- **Greedy answers on 6 long prompts:** 100% matched on the first pass, 83% from the prefix cache (needed 67%).
- **Control:** turning the 4,096 limit off moves plain Transformers by 2.43, so a vLLM ignoring the limit would fail this check. Why it moves that much wasn't investigated; the test never turns the limit off.
- **`window` vs the earlier plain Transformers run, all 618 answers:** average gap 0.0025, signed +0.00005, largest 0.0182. The window's loss, 2.609, reproduces the earlier 2.609.
- **Prefix cache, 40 long prompts, 16 greedy tokens, first-token log-probs:**

| Comparison | All 16 tokens same | Average log-prob gap on shared tokens | Signed |
| --- | --- | --- | --- |
| Repeat, no cache | 34/40 | 0.0079 | +0.00026 |
| Different batches, no cache | 31/40 | 0.0141 | +0.00173 |
| From cache vs from scratch | 34/40 | 0.0119 | +0.00008 |

The cache behaves like repeat noise. It served about 26% of tokens in that test, and 49% in the generation sample below. Runs aren't word-for-word repeatable, but they don't lean one way.

## Loss on my real answer (lower is better)

Paired answer by answer against the window; 95% intervals resample whole classes.

| Version | Loss | vs window | 95% interval | Better than window on |
| --- | --- | --- | --- | --- |
| `window` | 2.6090 |  |  |  |
| `tail` | 2.5279 | -0.0812 | -0.092 to -0.071 | 82% |
| `full` | 2.5145 | -0.0945 | -0.106 to -0.083 | 82% |
| `full-top` | 2.5253 | -0.0837 | -0.093 to -0.074 | 82% |
| `full-tag` | 2.5374 | -0.0716 | -0.084 to -0.061 | 76% |
| `full-pcw` | 2.5139 | -0.0951 | -0.107 to -0.084 | 83% |

For scale: models A and B differ by 0.029, and the decision limit was +0.015.

**By length of the full prompt + answer, vs window:**

| Length | Answers (classes) | `tail` | `full` | `full-top` | `full-tag` |
| --- | --- | --- | --- | --- | --- |
| up to 4,096 | 141 (65) | -0.0680 (-0.097 to -0.038) | -0.0698 (-0.098 to -0.041) | -0.0653 (-0.088 to -0.044) | -0.0588 (-0.087 to -0.030) |
| 4,097-8,192 | 195 (66) | -0.1016 (-0.123 to -0.083) | -0.1166 (-0.140 to -0.093) | -0.1010 (-0.118 to -0.083) | -0.0900 (-0.112 to -0.069) |
| over 8,192 | 282 (71) | -0.0736 (-0.086 to -0.061) | -0.0916 (-0.104 to -0.079) | -0.0809 (-0.091 to -0.070) | -0.0653 (-0.081 to -0.050) |

`full-pcw` wasn't split by length.

## Does it still sound like me?

| Measure | `window` | `tail` | `full` | `full-top` | `full-tag` | `full-pcw` | Me |
| --- | --- | --- | --- | --- | --- | --- | --- |
| "like" per 1,000 words | 58.9 | 60.5 | 60.0 | 63.6 | 58.3 | 60.1 | 44.7 |
| "I think" per 1,000 words | 10.6 | 11.1 | 10.9 | 12.7 | 10.6 | 11.0 | 5.9 |
| "So at sentence start" per 1,000 words | 13.6 | 12.1 | 13.7 | 14.7 | 12.5 | 13.5 | 11.8 |
| "yeah / yep" per 1,000 words | 6.0 | 5.3 | 7.8 | 9.4 | 6.4 | 9.9 | 6.8 |
| Median words | 61 | 65 | 75 | 53 | 75 | 71 | 55 |
| Mean words | 77.2 | 78.8 | 84.5 | 67.5 | 88.2 | 82.0 | 80.8 |
| Cut off at the length limit | 7% | 8% | 9% | 5% | 11% | 8% |  |
| Answers stuck in a loop | 81 | 60 | 55 | 44 | 64 | 54 |  |
| Tried to write another speaker | 0 | 0 | 0 | 0 | 0 | 0 |  |
| Content-word overlap with my answer | 8.9% | 9.8% | 9.8% | 8.9% | 10.2% | 10.1% |  |
| ...with an unrelated answer of mine | 4.2% | 4.2% | 4.1% | 3.2% | 4.1% | 4.1% |  |
| Closer in meaning to my answer than to an unrelated one | 78% | 83% | 83% | 81% | 82% | 84% |  |

**Habits, how much further from my rate than the window, per 1,000 words (95% interval; above 0 means further):**

| Habit | `tail` | `full` | `full-top` | `full-tag` | `full-pcw` |
| --- | --- | --- | --- | --- | --- |
| like | +1.55 (-2.99 to +5.90) | +1.01 (-3.04 to +4.79) | +4.64 (+0.17 to +8.96) | -0.60 (-5.09 to +3.44) | +1.16 (-2.53 to +4.71) |
| I think | +0.45 (-1.34 to +2.12) | +0.25 (-1.59 to +2.03) | +2.08 (+0.26 to +3.82) | -0.05 (-1.93 to +1.85) | +0.41 (-1.58 to +2.38) |
| So at sentence start | -1.51 (-2.60 to -0.06) | +0.04 (-1.55 to +1.62) | +1.10 (-0.57 to +2.85) | -1.09 (-2.36 to +0.58) | -0.09 (-1.56 to +1.46) |
| yeah / yep | +0.75 (-0.28 to +1.69) | +0.32 (-1.05 to +2.11) | +1.87 (+0.16 to +3.58) | -0.36 (-1.17 to +0.69) | +2.43 (-0.77 to +7.26) |

"like" and "I think" are already too high on the window, about 1.3x and 1.8x my rate. That comes from training, and class context doesn't fix it.

**Answer length vs the window (ratio, 95% interval; the limit is about 10%):**

| Version | Ratio | 95% interval |
| --- | --- | --- |
| `tail` | 1.02x | 0.94x to 1.10x |
| `full` | 1.12x | 1.05x to 1.21x |
| `full-top` | 0.90x | 0.83x to 0.97x |
| `full-tag` | 1.21x | 1.11x to 1.31x |
| `full-pcw` | 1.07x | 1.00x to 1.15x |

**Spread of answer length, in words:**

|  | 10th pct | 25th | Median | 75th | 90th | Mean | Under 30 words | Over 150 words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Me | 11 | 24 | 55 | 119 | 180 | 80.8 | 30% | 17% |
| `window` | 13 | 26 | 61 | 114 | 170 | 77.2 | 28% | 15% |
| `tail` | 13 | 28 | 65 | 116 | 185 | 78.8 | 27% | 16% |
| `full` | 15 | 33 | 75 | 126 | 184 | 84.5 | 22% | 18% |
| `full-top` | 13 | 26 | 53 | 97 | 144 | 67.5 | 29% | 8% |
| `full-tag` | 17 | 36 | 75 | 129 | 187 | 88.2 | 20% | 21% |
| `full-pcw` | 13 | 30 | 71 | 122 | 180 | 82.0 | 24% | 17% |

`full` isn't too long across the board: its long answers almost match mine, and it gives too few short ones. So a uniform stop boost would shorten my long answers too. The 256-token limit caps the top end, at about 190 words; 55 of my real answers are longer than that.

**Median words by prompt length (share cut off in brackets):**

| Length | Answers | Me | `window` | `tail` | `full` | `full-top` | `full-tag` | `full-pcw` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| up to 4,096 | 141 | 52 | 65 (6%) | 62 (7%) | 58 (8%) | 50 (4%) | 57 (6%) | 59 (6%) |
| 4,097-8,192 | 195 | 51 | 56 (4%) | 68 (7%) | 76 (9%) | 49 (5%) | 75 (10%) | 67 (7%) |
| over 8,192 | 282 | 65 | 65 (9%) | 62 (9%) | 78 (9%) | 60 (5%) | 84 (13%) | 77 (11%) |

## Placement: what can every layer see?

| Within the last 4,096 tokens | `full` | `full-top` | `tail` |
| --- | --- | --- | --- |
| The voice tag | 149 of 618 | 618 of 618 | 618 of 618 |
| The ask and replies | All | All | All |
| My last line in the prompt, where there is one | 514 of 554 | 496 of 554 | 465 of 465 |
| Median tokens from my last line to the end | 496 | 1,170 | 294 |

64 prompts have no earlier line of mine. Out-of-reach text is still read by the 8 full layers. Keeping the tag in reach didn't help: `full` beats `full-top` on loss, with habits no further from mine than the window.

**Direct comparisons on loss (95% interval; negative favours the first):**

| Comparison | Answers | Difference | 95% interval |
| --- | --- | --- | --- |
| `full` minus `tail` | 618 | -0.0134 | -0.0181 to -0.0088 |
| `full` minus `full-top` | 618 | -0.0108 | -0.0175 to -0.0046 |
| `full-top` minus `tail` | 618 | -0.0026 | -0.0091 to +0.0046 |
| `full-tag` minus `full` | 618 | +0.0229 | +0.0176 to +0.0283 |
| `full-tag` minus `full-top` | 618 | +0.0121 | +0.0050 to +0.0193 |
| `full-tag` minus `tail` | 618 | +0.0096 | +0.0042 to +0.0148 |

## Pre-class work

333 of the 618 prompts have pre-class work, from 45 of the 84 classes. The 300-word cut removes part of the answers on about 200 of those, and on average the prompts keep 77% of the answer words.

**Does it need all 32 layers?** `full` leaves it with the 8 full layers on long prompts, and `full-top` keeps it in reach of all 32. If it needed all 32, `full` would fall behind where pre-class work exists. These groups aren't randomly assigned, so this is support rather than proof:

| Prompts over 4,096 tokens | Answers | `full` minus `full-top` | 95% interval |
| --- | --- | --- | --- |
| With pre-class work | 275 | -0.0131 | -0.0212 to -0.0053 |
| Without | 202 | -0.0121 | -0.0203 to -0.0051 |

| Prompts over 4,096 tokens | Answers | `full-tag` minus `full-top` | 95% interval |
| --- | --- | --- | --- |
| With pre-class work | 275 | +0.0182 | +0.0070 to +0.0294 |
| Without | 202 | +0.0077 | -0.0013 to +0.0148 |

**All of it instead of the first 300 words (`full-pcw` vs `full`):**

| Where the text changed | `full` | `full-pcw` | Difference (95% interval) |
| --- | --- | --- | --- |
| Loss vs window (299 answers, 40 classes) | -0.0940 | -0.0951 | -0.0012 (-0.0028 to +0.0001) |
| Median words (mine 51, window 49) | 71 | 63 | 0.91x (0.79x to 1.02x) |
| "like" per 1,000 (mine 45.3) | 56.8 | 57.0 | further from mine by +0.12 (-6.11 to +6.10) |
| "I think" per 1,000 (mine 5.1) | 9.6 | 9.9 | further from mine by +0.28 (-1.66 to +2.33) |
| "So at sentence start" per 1,000 (mine 13.8) | 15.5 | 15.3 | further from mine by -0.18 (-2.40 to +1.88) |
| "yeah / yep" per 1,000 (mine 7.6) | 9.1 | 14.0 | further from mine by +4.86 (-1.23 to +14.88) |
| Content-word overlap with my answer | 9.8% | 10.3% |  |
| Meaning similarity to my answer | 0.695 | 0.688 | (-0.019 to +0.005) |
| Cut off at the length limit | 8% | 7% |  |
| Answers stuck in a loop | 21 | 20 |  |

| Words added to the prompt | Answers (classes) | Loss vs `full` | 95% interval |
| --- | --- | --- | --- |
| added under 50 words | 50 (10) | +0.0001 | -0.001 to +0.001 |
| added 50-299 words | 111 (14) | +0.0002 | -0.002 to +0.002 |
| added 300+ words | 138 (16) | -0.0027 | -0.005 to -0.000 |

No measurable difference: at most a very small gain where the most text was added. It costs a median of 239 tokens. No version removed pre-class work, so this doesn't show what it adds in itself. On long prompts the extra text reaches only the 8 full layers.

## Blind picks

4 screens so far, on `window`, `tail` and `full`, shuffled and unlabelled, on questions not judged before. **Too few to count:** about 20 are needed, and a version needs about 11 best picks of 20 to stand out from chance.

| Version | Picked as best | Sounds like me |
| --- | --- | --- |
| `window` | 2 of 4 | 2 of 4 |
| `tail` | 0 of 4 | 3 of 4 |
| `full` | 2 of 4 | 3 of 4 |

## Speed

One tenth of the real scoring job (1,246,549 tokens) under each setting, on real prompts. Scoring differences are compared with the first row:

| Setting | Time | Tokens/s | Full job | Average score shift | Signed | Largest |
| --- | --- | --- | --- | --- | --- | --- |
| FlashAttention 2, existing settings | 58.4s | 21,345 | 9.7 min |  |  |  |
| Same, decoding off | 59.2s | 21,057 | 9.8 min | 0.00002 | +0.00002 | 0.00231 |
| Batches of 16,384 tokens (a third less KV room) | 59.1s | 21,092 | 9.8 min | 0.00210 | +0.00014 | 0.01501 |
| Triton attention | 73.6s | 16,937 | 12.2 min | 0.00209 | -0.00007 | 0.01113 |

The existing settings were kept. FlashInfer can't run this model: it raises an error when layers mix sliding and full attention. Generation: 316 prompts in 76 seconds, with 49% of prompt tokens from the prefix cache. The four-version run took 21 minutes in all, against an unmeasured estimate of 1.5-2 hours.

## What this doesn't show

- **That retraining couldn't do better.** It shows only that the model already uses the whole class without one.
- **That voice is settled.** The evidence so far is habit counts, length and 4 blind screens.
- **How it works live.** Every prompt came from a moment when I actually spoke, and a live bot also has to decide when to speak. First-word delay on prompts this long is unmeasured.
- **`full-tag` and `full-pcw` had no vLLM check of their own.** They use the same code path at about the same lengths. Each was also a separate run from `full`, so run-to-run noise (about 0.002 per answer, signed about 0.0001) is part of their differences.
