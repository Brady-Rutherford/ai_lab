#!/usr/bin/env bash
# On the POD: try DPO settings that counter length drift on 1B model A, and keep the one whose
# answers stay closest to my length. eval_drift.py and eval_memorization.py must pass. Writes the
# winner's train_dpo.py flags to ~/dpo_flags; with no winner the file is absent, and the 7B gets
# no DPO.
#   bash ~/minerva-gpt/pod/dpo_sweep.sh
# Session 2: plain SPIN DPO gave answers 2.5x my length; skipping cut-off answers plus an SFT
# loss still gave 2.66x. The three settings below are the standard remedies.
set -uo pipefail
cd "$HOME/minerva-gpt"
declare -A CONFIGS=(
  [leash]="--beta 0.3 --sft-weight 2 --lr 2e-5"            # stay closer to the SFT model
  [lennorm]="--loss sigmoid_norm --sft-weight 1"            # per-token: length can't buy reward
  [lenmatch]="--length-match 0.3 --beta 0.2 --sft-weight 1" # winner and loser of similar length
)
rm -f "$HOME/dpo_flags"
best="" best_gap=""
for tag in leash lennorm lenmatch; do
  if SIZES=1b VARIANTS=A NO_RANK=1 TAG="$tag" DPO_FLAGS="${CONFIGS[$tag]}" bash pod/run_dpo_fix.sh; then
    ratio=$(grep -E "^drift: PASS" "$HOME/minerva.log" | tail -1 | sed -E 's/.*length ([0-9.]+)x mine.*/\1/')
    gap=$(python3 -c "print(abs($ratio - 1))")
    echo "== sweep: $tag passed, median length ${ratio}x mine" | tee -a "$HOME/minerva.log"
    if [ -z "$best" ] || python3 -c "import sys; sys.exit(not $gap < $best_gap)"; then
      best="$tag" best_gap="$gap"
    fi
  else
    echo "== sweep: $tag failed its checks" | tee -a "$HOME/minerva.log"
  fi
done
if [ -n "$best" ]; then
  echo "${CONFIGS[$best]}" > "$HOME/dpo_flags"
  echo "== sweep winner: $best (${CONFIGS[$best]})" | tee -a "$HOME/minerva.log"
else
  echo "== sweep: no setting passed; 7B stays SFT-only" | tee -a "$HOME/minerva.log"
fi
