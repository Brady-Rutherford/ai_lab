#!/usr/bin/env bash
# On the MAC: copy the code and the masked training data to the pod.
#   bash pod/push.sh root@HOST [PORT] [SSH_KEY]
# Code = the files git sees in experiments/minerva-gpt (no .venv, no nanochat, no checkpoints).
# Data = data/sft, data/sft_classmates, data/transcripts and stats.json: masked, and only after
# privacy_check.py passes. Once data/rankings.jsonl exists (the DPO session), the rankings,
# the session-1 adapters and the scored samples go up too.
# Never sent: raw CS156 data, the Forum cookie, any .env file.
set -euo pipefail
HOST="${1:?usage: push.sh user@host [port] [ssh_key]}"
PORT="${2:-22}"
KEY="${3:-}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new${KEY:+ -i $KEY}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

echo "== privacy gate: nothing leaves the Mac unless it finds 0 definite name hits"
(cd "$HERE" && uv run --project ~/cs156-ml python privacy_check.py data/sft data/sft_classmates \
  2>/dev/null | grep "privacy scan")

echo "== rsync on the pod (some images lack it)"
$SSH "$HOST" 'command -v rsync >/dev/null 2>&1 || { S=""; [ "$(id -u)" -ne 0 ] && S=sudo; \
  $S apt-get update -qq >/dev/null && $S env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null; }'

echo "== code"
(cd "$HERE" && git ls-files -co --exclude-standard . | grep -v -E '(^|/)\.env') > "$LIST"
rsync -az --files-from="$LIST" -e "$SSH" "$HERE/" "$HOST:minerva-gpt/"

echo "== data (masked)"
$SSH "$HOST" 'mkdir -p minerva-gpt/data'
rsync -az -e "$SSH" "$HERE/data/sft" "$HERE/data/sft_classmates" "$HERE/data/transcripts" \
  "$HERE/data/stats.json" "$HOST:minerva-gpt/data/"
if [ -f "$HERE/data/rankings.jsonl" ]; then
  echo "== for DPO (session 2): my rankings, the session-1 adapters and scored samples"
  rsync -az -e "$SSH" "$HERE/data/rankings.jsonl" "$HOST:minerva-gpt/data/"
  rsync -az -e "$SSH" "$HERE/data/generations" "$HOST:minerva-gpt/data/"
  rsync -az --exclude 'checkpoint-*' -e "$SSH" "$HERE/checkpoints" "$HOST:minerva-gpt/"
fi
echo "done. Next, on the pod: bash ~/minerva-gpt/pod/setup.sh"
