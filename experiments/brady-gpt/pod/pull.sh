#!/usr/bin/env bash
# On the MAC: bring the results back from the pod.
#   bash pod/pull.sh root@HOST [PORT] [SSH_KEY]
# Adapters (checkpoints/, without the trainer's checkpoint-* folders or merged final-* models), the grade
# model, the generations with their scores, and the log (into runs/). All gitignored.
set -euo pipefail
HOST="${1:?usage: pull.sh user@host [port] [ssh_key]}"
PORT="${2:-22}"
KEY="${3:-}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new${KEY:+ -i $KEY}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/checkpoints" "$HERE/data/generations" "$HERE/runs"
# final-* are merged full models (3 GB for 1B, ~15 GB for 7B): rebuilt on the Mac from the
# small adapters with merge_model.py when needed, so they stay on the pod
fail() { echo "!! $*"; echo "!! do NOT wipe the pod yet. Fix this and run pull.sh again."; exit 1; }
# A context-test pod has no checkpoints, and that must not stop the results below from coming back.
# (Inside $(...), an ssh that fails stops the script, rather than reading as "no checkpoints".)
has_checkpoints="$($SSH "$HOST" '[ -d brady-gpt/checkpoints ] && echo yes || echo no')"
if [ "$has_checkpoints" = yes ]; then
  rsync -az --exclude 'checkpoint-*' --exclude 'final-*' -e "$SSH" "$HOST:brady-gpt/checkpoints/" "$HERE/checkpoints/"
fi
rsync -az --exclude '*.running' -e "$SSH" "$HOST:brady-gpt/data/generations/" "$HERE/data/generations/" || true
rsync -az -e "$SSH" "$HOST:brady-gpt.log" "$HERE/runs/pod-$(date +%Y%m%d-%H%M).log" || true
du -sh "$HERE/checkpoints" "$HERE/data/generations"
# wipe.sh can't be undone. Every result on the pod must match this Mac's copy, by checksum: a file of
# the same name could be left over from an earlier run.
sums="$($SSH "$HOST" 'cd brady-gpt/data/generations && find . -type f -size +0c -exec sha256sum {} +')" \
  || fail "could not list the results on the pod"
if [ -n "$sums" ]; then
  bad="$(cd "$HERE/data/generations" && printf '%s\n' "$sums" | shasum -a 256 -c - 2>/dev/null | grep -v ': OK$' || true)"
  [ -z "$bad" ] || { printf '%s\n' "$bad"; fail "these results did not reach this Mac"; }
fi
running="$($SSH "$HOST" '[ -e brady-gpt/data/generations/7b/context-test.running ] && echo yes || echo no')"
[ "$running" = no ] || fail "the context test is still running or failed part way: see ~/context-test.log on the pod"
echo "done. Then, on the pod: bash ~/brady-gpt/pod/wipe.sh, and terminate it in the dashboard."
