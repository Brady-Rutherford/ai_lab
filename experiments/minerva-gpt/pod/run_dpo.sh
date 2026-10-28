#!/usr/bin/env bash
# On the POD, session 2: DPO for both variants on my rankings (pushed by pod/push.sh), plus
# automatic pairs if the voice score passed its check in session 1.
#   GATE_A=<a> GATE_B=<b> bash ~/minerva-gpt/pod/run_dpo.sh
# GATE_* are the "suggested --voice-gate" values session 1 printed. Leave a variant's gate
# unset to train it on my picks only, as it must be if its voice-score AUC check failed.
# pod/push.sh sends what it needs (rankings, checkpoints/sft-{A,B}, data/generations) once
# data/rankings.jsonl exists on the Mac.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/minerva-gpt"
[ -f data/rankings.jsonl ] || { echo "no data/rankings.jsonl: rank on the Mac, then push.sh"; exit 1; }
{
  for v in A B; do
    gate_var="GATE_$v"
    auto=()
    if [ -n "${!gate_var:-}" ]; then
      auto=(--auto "data/generations/$v-val.voice.grade.jsonl" --voice-gate "${!gate_var}")
    fi
    echo; echo "== $(date +%H:%M:%S) DPO, variant $v ${auto[*]:-(my picks only)}"
    uv run python train_dpo.py --variant "$v" --rankings data/rankings.jsonl "${auto[@]}"
  done
  echo; echo "== session 2 done. On the Mac: bash pod/pull.sh HOST PORT"
} 2>&1 | tee -a "$HOME/minerva.log"
