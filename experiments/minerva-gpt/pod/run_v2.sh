#!/usr/bin/env bash
# On the POD, session 2: retrain with the class context (readings, my pre-class work, the poll on
# screen), then DPO on SPIN pairs (my real answer vs the model's), for each variant. First on
# OLMo 2 1B; then, only if its memorization and regurgitation checks pass, on OLMo 3 7B.
# Logs to ~/minerva.log.
#   tmux new -s minerva          (detach: Ctrl-b then d; come back: tmux attach -t minerva)
#   bash ~/minerva-gpt/pod/run_v2.sh
# VARIANTS="A" limits it to one variant; SKIP_7B=1 stops after the 1B; START_AT=7b skips the
# 1B (after its checks were reviewed by hand, or redone with pod/run_dpo_fix.sh).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/minerva-gpt"
VARIANTS="${VARIANTS:-A B}"
START_AT="${START_AT:-1b}"
# Sized for an H200 (141 GB). The effective batch stays 32 (gradient accumulation makes up
# the rest), so bigger per-pass batches and no gradient checkpointing change speed, not
# results. On an 80 GB H100: BATCH_1B=8 BATCH_7B=2.
BATCH_1B="${BATCH_1B:-16}"
BATCH_7B="${BATCH_7B:-4}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # less fragmentation with ragged batches
BIG="allenai/Olmo-3-1025-7B"
FAILED=""
step() { echo; echo "== $(date +%H:%M:%S) $*"; }

# label, base model, SFT lr, DPO lr, batch per pass, extra SFT flags
one_model() {
  local label=$1 base=$2 sft_lr=$3 dpo_lr=$4 batch=$5 sft_flags=$6
  export MINERVA_BASE_MODEL="$base" MINERVA_CHECKPOINTS="checkpoints/$label"
  local C="$MINERVA_CHECKPOINTS" G="data/generations/$label"
  mkdir -p "$G"
  for v in $VARIANTS; do
    step "[$label] SFT $v with class context"
    uv run python train_sft.py --variant "$v" --lr "$sft_lr" --batch "$batch" $sft_flags
    if [ "$v" = B ]; then
      step "[$label] B: does the voice score tell my real answers from classmates'?"
      uv run python voice_score.py --adapter "$C/sft-B" --variant B --validate val
    fi
    step "[$label] $v: answers to train asks (for SPIN) and never-seen asks, no retrieval"
    uv run python generate.py --adapter "$C/sft-$v" --variant "$v" --split train --source all --n 1 --engine vllm --out "$G/$v-train.jsonl"
    uv run python generate.py --adapter "$C/sft-$v" --variant "$v" --split val --source all --n 1 --engine vllm --out "$G/$v-val-plain.jsonl"
    step "[$label] $v: memorization after SFT"
    uv run python eval_memorization.py "$G/$v-train.jsonl" "$G/$v-val-plain.jsonl" || FAILED="$FAILED $label-$v-sft-memorization"
    step "[$label] $v: DPO on SPIN pairs, then fold SFT + DPO into one model"
    uv run python train_dpo.py --variant "$v" --spin "$G/$v-train.jsonl" --lr "$dpo_lr"
    uv run python merge_model.py "$C/sft-$v" "$C/dpo-$v" --out "$C/final-$v"
    step "[$label] $v: memorization after DPO (it trained on those answers again)"
    uv run python generate.py --adapter "$C/final-$v" --variant "$v" --split train --source all --n 1 --limit 600 --engine vllm --out "$G/$v-train-final.jsonl"
    uv run python generate.py --adapter "$C/final-$v" --variant "$v" --split val --source all --n 1 --limit 600 --engine vllm --out "$G/$v-val-final-plain.jsonl"
    uv run python eval_memorization.py "$G/$v-train-final.jsonl" "$G/$v-val-final-plain.jsonl" || FAILED="$FAILED $label-$v-dpo-memorization"
    uv run python eval_drift.py "$G/$v-val-final-plain.jsonl" || FAILED="$FAILED $label-$v-dpo-drift"
    step "[$label] $v: answers to rank and blind-test (final model, retrieval on)"
    uv run python generate.py --adapter "$C/final-$v" --variant "$v" --source all --n 2 --retrieve --engine vllm --out "$G/$v-val.jsonl"
  done
  step "[$label] regurgitation: does either model repeat someone else's words?"
  uv run python eval_regurgitation.py "$G"/*-val.jsonl || FAILED="$FAILED $label-regurgitation"
}

{
  if [ "$START_AT" = 1b ]; then
    one_model 1b "allenai/OLMo-2-0425-1B" 2e-4 5e-5 "$BATCH_1B" "--no-checkpointing"
  fi
  if [ -n "$FAILED" ]; then
    step "1B checks failed:$FAILED. Stopping before 7B; look at the log first."
  elif [ -n "${SKIP_7B:-}" ]; then
    step "1B done, SKIP_7B set."
  else
    step "Fetching $BIG and checking vLLM can serve it"
    uv run python download_model.py "$BIG"
    MINERVA_BASE_MODEL="$BIG" "$HOME/vllm-env/bin/python" -c \
      "import common; from vllm import LLM; LLM(common.BASE_MODEL, max_model_len=2048).generate('Hello'); print('vLLM serves', common.BASE_MODEL)"
    # 7B: a lower learning rate; the H200 has the memory to skip gradient checkpointing
    one_model 7b "$BIG" 1e-4 2.5e-5 "$BATCH_7B" "--no-checkpointing"
    [ -n "$FAILED" ] && step "7B checks failed:$FAILED"
  fi
  step "session 2 done. On the Mac: bash pod/pull.sh HOST PORT"
} 2>&1 | tee -a "$HOME/minerva.log"
