#!/usr/bin/env bash
# On the MAC: bring the results back from the pod.
#   bash pod/pull.sh root@HOST [PORT] [SSH_KEY]
# Adapters (checkpoints/, without the trainer's intermediate checkpoint-* folders), the grade
# model, the generations with their scores, and the log (into runs/). All gitignored.
set -euo pipefail
HOST="${1:?usage: pull.sh user@host [port] [ssh_key]}"
PORT="${2:-22}"
KEY="${3:-}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new${KEY:+ -i $KEY}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/checkpoints" "$HERE/data/generations" "$HERE/runs"
rsync -az --exclude 'checkpoint-*' -e "$SSH" "$HOST:minerva-gpt/checkpoints/" "$HERE/checkpoints/"
rsync -az -e "$SSH" "$HOST:minerva-gpt/data/generations/" "$HERE/data/generations/" || true
rsync -az -e "$SSH" "$HOST:minerva.log" "$HERE/runs/pod-$(date +%Y%m%d-%H%M).log" || true
du -sh "$HERE/checkpoints" "$HERE/data/generations"
echo "done. Then, on the pod: bash ~/minerva-gpt/pod/wipe.sh, and terminate it in the dashboard."
