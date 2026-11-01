#!/usr/bin/env bash
# On the POD, the rest of session 2 after the 1B DPO drifted: pick a DPO setting on 1B, train the
# 7B SFT-only (checks plus answers to rank), then 7B DPO only with a setting that passed.
#   tmux new -s minerva; bash ~/minerva-gpt/pod/run_session2b.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/minerva-gpt"
bash pod/dpo_sweep.sh
START_AT=7b DPO=0 bash pod/run_v2.sh
if [ -f "$HOME/dpo_flags" ]; then
  SIZES=7b TAG=dpobest DPO_FLAGS="$(cat "$HOME/dpo_flags")" bash pod/run_dpo_fix.sh
else
  echo "== no DPO setting passed on 1B: the 7B stays SFT-only" | tee -a "$HOME/minerva.log"
fi
echo "== session 2b done. On the Mac: bash pod/pull.sh HOST PORT" | tee -a "$HOME/minerva.log"
