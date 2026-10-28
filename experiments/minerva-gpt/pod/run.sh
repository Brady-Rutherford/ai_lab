#!/usr/bin/env bash
# On the POD, session 1: SFT both variants, check the voice score and the grade model, sample
# answers for me to rank (vLLM), score them, and run the regurgitation test. Logs to ~/minerva.log.
#   tmux new -s minerva          (detach: Ctrl-b then d; come back: tmux attach -t minerva)
#   bash ~/minerva-gpt/pod/run.sh
# Same code path as the Mac smoke test (sdpa attention, no packing); a big GPU just runs it with
# bigger batches and without gradient checkpointing, which changes speed, not results.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/minerva-gpt"
BATCH="${BATCH:-8}"
step() { echo; echo "== $(date +%H:%M:%S) $*"; }
{
  step "SFT, variant A (my answers only)"
  uv run python train_sft.py --variant A --batch "$BATCH" --no-checkpointing
  step "SFT, variant B (mine x3 + classmates x1, tagged)"
  uv run python train_sft.py --variant B --batch "$BATCH" --no-checkpointing

  step "voice score: does it tell my real answers from classmates'? (AUC, suggested gate)"
  uv run python voice_score.py --adapter checkpoints/sft-B --variant B --validate val
  uv run python voice_score.py --adapter checkpoints/sft-A --variant A --validate val

  step "grade model: Spearman gate and label-dropping test"
  uv run python reward_model.py --train

  step "sampling with vLLM: 2 answers per held-out ask from each variant, retrieval on"
  for v in A B; do
    uv run python generate.py --adapter "checkpoints/sft-$v" --variant "$v" --source all \
      --n 2 --retrieve --engine vllm
  done

  step "scoring the samples (voice, then grade)"
  for v in A B; do
    uv run python voice_score.py --adapter "checkpoints/sft-$v" --variant "$v" \
      --score "data/generations/$v-val.jsonl"
    uv run python reward_model.py --score "data/generations/$v-val.voice.jsonl"
  done

  step "regurgitation: does either model repeat someone else's words?"
  uv run python eval_regurgitation.py data/generations/A-val.jsonl data/generations/B-val.jsonl \
    || echo "!! REGURGITATION HITS above: do not use those samples until we look at them"

  step "session 1 done. On the Mac: bash pod/pull.sh HOST PORT"
} 2>&1 | tee -a "$HOME/minerva.log"
