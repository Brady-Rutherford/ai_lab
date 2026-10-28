#!/usr/bin/env bash
# On the MAC: get the code onto the pod and copy the masked training data to it.
#   bash pod/push.sh root@HOST [PORT] [SSH_KEY]
# Code = cloned on the pod from GitHub (public repo, branch $BRANCH, default main) into ~/ai_lab,
# with ~/minerva-gpt linking to the experiment. It refuses if this folder differs from GitHub,
# so the pod never runs code you haven't pushed. CODE=mac sends this Mac's files instead.
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
REPO="${REPO:-https://github.com/Brady-Rutherford/ai_lab.git}"
BRANCH="${BRANCH:-main}"
CODE="${CODE:-github}"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

echo "== privacy gate: nothing leaves the Mac unless it finds 0 definite name hits"
(cd "$HERE" && uv run --project ~/cs156-ml python privacy_check.py data/sft data/sft_classmates \
  2>/dev/null | grep "privacy scan")

echo "== rsync and git on the pod (some images lack them)"
$SSH "$HOST" 'command -v rsync >/dev/null 2>&1 && command -v git >/dev/null 2>&1 || { S=""; [ "$(id -u)" -ne 0 ] && S=sudo; \
  $S apt-get update -qq >/dev/null && $S env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync git >/dev/null; }'

if [ "$CODE" = github ]; then
  echo "== code: GitHub $BRANCH, after checking it matches this Mac"
  git -C "$HERE" fetch -q origin "$BRANCH"
  if ! git -C "$HERE" diff --quiet "origin/$BRANCH" -- . \
     || [ -n "$(git -C "$HERE" ls-files -o --exclude-standard .)" ]; then
    echo "!! this folder differs from GitHub $BRANCH (uncommitted, unpushed or unmerged changes):"
    git -C "$HERE" diff --stat "origin/$BRANCH" -- . | tail -5
    echo "   The pod would run GitHub's version. Push and merge first, or send this Mac's code:"
    echo "   CODE=mac bash pod/push.sh $HOST $PORT"
    exit 1
  fi
  $SSH "$HOST" "if [ -d ai_lab/.git ]; then git -C ai_lab fetch -q origin $BRANCH && git -C ai_lab checkout -q $BRANCH \
      && git -C ai_lab reset -q --hard origin/$BRANCH; else git clone -q --branch $BRANCH $REPO ai_lab; fi \
    && ln -sfn ~/ai_lab/experiments/minerva-gpt ~/minerva-gpt && git -C ai_lab log --oneline -1"
else
  echo "== code: this Mac's files"
  (cd "$HERE" && git ls-files -co --exclude-standard . | grep -v -E '(^|/)\.env') > "$LIST"
  rsync -az --files-from="$LIST" -e "$SSH" "$HERE/" "$HOST:minerva-gpt/"
fi

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
