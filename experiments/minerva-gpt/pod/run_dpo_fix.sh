#!/usr/bin/env bash
# On the POD, after run_v2.sh: redo only the DPO step, with the length fix, for the 1B and 7B
# models whose SFT adapters run_v2.sh already trained. Logs to ~/minerva.log.
#   tmux new -s dpofix; bash ~/minerva-gpt/pod/run_dpo_fix.sh
# Session 2 found DPO pushing answers from 75 to 191 words (66% cut off at the cap): cut-off
# model answers ended with eos in the pairs, so "stopping" was marked as losing. The fix: sample
# with finish flags, skip cut-off answers in the SPIN pairs, and add an SFT loss on my real
# answers (RPO). eval_drift.py then gates the shape of the answers.
# SIZES="1b" or VARIANTS="A" narrow it down. DPO_FLAGS passes settings to train_dpo.py; TAG
# names the outputs (checkpoints/<size>/<TAG>-A, final-<TAG>-A, <variant>-val-<TAG>.jsonl).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/minerva-gpt"
SIZES="${SIZES:-1b 7b}"
VARIANTS="${VARIANTS:-A B}"
DPO_FLAGS="${DPO_FLAGS:-}"
TAG="${TAG:-dpo2}"
FAILED=""
step() { echo; echo "== $(date +%H:%M:%S) $*"; }
base_of() { [ "$1" = 7b ] && echo "allenai/Olmo-3-1025-7B" || echo "allenai/OLMo-2-0425-1B"; }
dpo_lr_of() { [ "$1" = 7b ] && echo 2.5e-5 || echo 5e-5; }

{
  for label in $SIZES; do
    export MINERVA_BASE_MODEL="$(base_of "$label")" MINERVA_CHECKPOINTS="checkpoints/$label"
    C="$MINERVA_CHECKPOINTS" G="data/generations/$label"
    for v in $VARIANTS; do
      [ -d "$C/sft-$v" ] || { step "[$label] no $C/sft-$v, skipping"; continue; }
      if [ ! -f "$G/$v-train-v2.jsonl" ]; then
        step "[$label] $v: SFT answers to train asks, with finish flags"
        uv run python generate.py --adapter "$C/sft-$v" --variant "$v" --split train --source all --n 1 --engine vllm --out "$G/$v-train-v2.jsonl"
      fi
      step "[$label] $v: DPO [$TAG] ${DPO_FLAGS:-(defaults)}"
      uv run python train_dpo.py --variant "$v" --spin "$G/$v-train-v2.jsonl" --lr "$(dpo_lr_of "$label")" --out "$C/$TAG-$v" $DPO_FLAGS
      uv run python merge_model.py "$C/sft-$v" "$C/$TAG-$v" --out "$C/final-$TAG-$v"
      step "[$label] $v: drift and memorization [$TAG]"
      uv run python generate.py --adapter "$C/final-$TAG-$v" --variant "$v" --split val --source all --n 1 --limit 600 --engine vllm --out "$G/$v-val-$TAG-plain.jsonl"
      uv run python eval_drift.py "$G/$v-val-$TAG-plain.jsonl" || FAILED="$FAILED $label-$v-drift"
      uv run python generate.py --adapter "$C/final-$TAG-$v" --variant "$v" --split train --source all --n 1 --limit 600 --engine vllm --out "$G/$v-train-$TAG.jsonl"
      uv run python eval_memorization.py "$G/$v-train-$TAG.jsonl" "$G/$v-val-$TAG-plain.jsonl" || FAILED="$FAILED $label-$v-memorization"
      [ -n "${NO_RANK:-}" ] && continue
      step "[$label] $v: answers to rank and blind-test [$TAG] (retrieval on)"
      uv run python generate.py --adapter "$C/final-$TAG-$v" --variant "$v" --source all --n 2 --retrieve --engine vllm --out "$G/$v-val-$TAG.jsonl"
    done
    if [ -z "${NO_RANK:-}" ] && compgen -G "$G/*-val-$TAG.jsonl" >/dev/null; then
      step "[$label] regurgitation [$TAG]"
      uv run python eval_regurgitation.py "$G"/*-val-"$TAG".jsonl || FAILED="$FAILED $label-regurgitation"
    fi
  done
  step "DPO [$TAG] done.${FAILED:+ Failed:$FAILED}"
  [ -z "$FAILED" ]
} 2>&1 | tee -a "$HOME/minerva.log"
