# Plan: the model takes part in one live class (planned, not started)

The 7B model (variant B, SFT + DPO) joins one online CS156 session as me: it listens, decides
when to raise its hand, and answers aloud in my cloned voice, camera off. This is a CS156 project
demo. **The professor has agreed that it will happen in one session** (without knowing which
one), and it is revealed to the class in that same session.

Status: planned on 2 Nov 2026. Nothing below is built yet.

## What already exists

- **The model.** `bradyrutherford/minerva-olmo3-7b-B` (private): full bf16, SFT + DPO. It won
  the comparisons on my spoken answers (held-out loss, word mix, my blind picks), and its 8-bit
  MLX copy runs locally with `chat.py`.
- **Training data from the same transcriber.** The 477 class transcripts come from Deepgram
  (Scribe v3), so a live Deepgram stream produces text in the format the model learned from.
- **On Cue** (`~/cs156-ml/pipeline1`): 2,774 of my hand raises timed from the recordings
  (559 in the 94 test classes), and a finished hand-raise model (MLP + TCN). On test it is real
  but weak: 2.3-3x random; at its threshold it fires about 6 times per class, 7% of firings
  land 2-15 s before a real raise, and it warns ahead of about 7% of raises (35 of 533). It can
  rank moments, but it can't be the only trigger.

## Architecture: split, with the heavy part on a GPU

| Runs on | What |
|---|---|
| **GPU pod** | vLLM serving the full bf16 model with prefix caching (the class context stays cached); On Cue's scorer |
| **Mac** | Forum in my normal browser; audio routing (BlackHole); a thin client: Deepgram stream, policy, ElevenLabs, dashboard, stop key |

The two talk over an SSH tunnel (`ssh -N -L 8000:127.0.0.1:8000 pod`). The model server is
never exposed. The Mac's load is about what a video call takes.

**Not chosen (yet): fully in the cloud**, a VM joining Forum in a headless browser with virtual
audio. It is hands-off, but Forum may reject the session from a datacenter IP, headless meeting
audio is fiddly, and the Deepgram and ElevenLabs keys would have to live on the pod. Revisit
only if the split version works.

## The constrained cloud environment

- **Inbound:** SSH only, key auth. vLLM binds to `127.0.0.1`, reachable only through the tunnel.
- **Secrets:** the HF token is piped into the download process only, never written on the pod.
  Deepgram and ElevenLabs keys stay on the Mac.
- **Data on the pod:** the weights, that session's context (readings, my pre-class work) and
  transcript snippets as they arrive. Logs are written back to the Mac.
- **Outbound:** needed for the HF download during setup; can be closed in the provider's
  firewall afterwards.
- **Disposable:** `pod/wipe.sh`, then terminate.
- **Pinned:** the vLLM environment from `pod/setup.sh` (FlashInfer sampler off).

## GPU

The model is 14.6 GB, so 24 GB is enough. Decoding speed follows memory bandwidth (rough):

| GPU | Approx. decode | Note |
|---|---|---|
| L4 (24 GB) | ~15 tok/s | cheapest; still faster than speech (~3-4 tok/s) |
| A10 / A10G (24 GB) | ~30 tok/s | fits |
| L40S (48 GB) / A100 | ~40-75 tok/s | preferred: headroom, fast replays |
| H100 | ~150 tok/s | more than needed |

With the context cached, the first word should come about 1 s after a question ends (about
3-4 s on the Mac).

## The bot, part by part

1. **Ears.** Forum audio → BlackHole → Deepgram streaming (no-retention API). Diarized speakers
   map to roles; the one who talks most is the professor.
2. **Context: the `full` prompt (decided 4 Nov 2026; results in `NEXT_SESSION.md`).** Before class, fetch
   the session header, focused LOs/HCs, readings and my pre-class work with the existing Forum
   harvester. Live: a `Class so far:` block with everything said up to that moment, then the
   professor's latest ask plus the replies since, under the same word budgets as `prepare_data.py`.
   The model was trained on short prompts, but the context test found the whole class predicts my
   answers better at every length, with no retraining. Keep this exact layout: moving the class or
   adding lines changed the results.
3. **Brain.** Prompts go to vLLM as the same token ids as training (`generate.prompt_ids`).
   Drafting starts the moment the professor finishes a question, so an answer is always ready.
4. **When to speak.** v1: only after open questions, capped at my own rate (about 6 raises per
   class), spaced out, and only when the draft passes the guardrails. v2: On Cue's score ranks
   the moments. If the professor says "Brady", it always answers.
5. **Voice.** ElevenLabs streaming TTS with my cloned voice → a second virtual audio device →
   Forum's mic input. Press `H` to raise the hand, unmute when called, mute again after.
   The clone is built only from clips of my own turns in the recordings.
6. **Guardrails.** Answers capped at about 60-80 words; drafts that loop ("he's he's he's") or
   contain `[NAME]` are thrown away and redrafted; a hotkey stops everything at once; a local
   dashboard shows the transcript and every decision.
7. **Log.** Timestamped transcript, scores, decisions, drafts and latencies, for the write-up.

## Phases

1. **Replay engine.** Feed a held-out test class's transcript, in real time or faster, through
   parts 2-4 and 7, against the pod. Score it against what I actually did: hand-raise timing vs
   my 559 real raises (hits, misses, false alarms), a blind test of its answers vs mine at the
   same moments, and latency. Fully offline apart from the pod.
2. **On Cue live.** Plug the scorer in. Its features came from Forum's event logs, which aren't
   all visible live, so it may need retraining on transcript and timing features only. That
   comparison is a section of its own.
3. **Live input/output.** Audio routing, Deepgram streaming, ElevenLabs, Forum keys. Dry run in
   a private Zoom or Meet.
4. **Dress rehearsal.** A recorded test class's audio played into the full live system.
5. **The class.** One activity block (20-30 minutes), capped at about 3-4 contributions, then the
   reveal. A whole class next time if the first one goes well.

## Open decisions

- Provider and GPU (L40S or A100 preferred).
- Whether v1 ships with On Cue or with the rule-based timing only.
- Sampling temperature: 0.8 so far. The models overuse my habits ("like" about 1.5x my rate);
  `runs/temp_test.py` compares 0.8, 0.9 and 1.0 (29 of 40 questions done).

## Known risks

- Without class context the model can drift or loop; live prompts must always carry the class.
- `full` prompts reach about 19,000 tokens by the end of a class. How much vLLM's prefix cache saves at
  that length, and the first-word delay, are unmeasured: check both in the replay phase.
- `full` answers run longer than mine (median 75 words vs 55), and "like" and "I think" stay too high.
- It states made-up facts about me with confidence; keep it to class discussion.
- Speaker roles from diarization can be wrong when the professor speaks briefly.
- A weak hand-raise model means odd timing: rules and caps come first, the model second.
