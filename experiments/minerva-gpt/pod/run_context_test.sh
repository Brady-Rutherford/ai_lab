#!/usr/bin/env bash
# On the POD: the context test (eval_context.py) on the chosen model, variant B with SFT + DPO.
# Does it answer as well with the class so far in its prompt? See NEXT_SESSION.md for the commands.
#
# Steps: download and checksum the model; check vLLM against plain Transformers on long prompts
# (this also picks the attention backend, and stops the run if vLLM is wrong); score; generate.
# A rerun skips whatever is done, so after a crash just run it again.
#
# The HF login is needed only for the download, and is dropped right after it:
#   ssh -p PORT USER@HOST 'cd ~/minerva-gpt && HF_TOKEN=$(cat) LIMIT=30 bash pod/run_context_test.sh' \
#     < ~/.cache/huggingface/token
# Options: LIMIT=30 (a quick pass; its results go to data/generations/7b/quick), TAIL=2500 (tokens of
# the class for `tail`), CROSSCHECK=30 (prompts compared), BACKEND=TRITON_ATTN (force an attention backend).
# Then pull.sh brings back data/generations/7b/context-test.json and the files beside it.
set -euo pipefail
cd "$HOME/minerva-gpt"
export PATH="$HOME/.local/bin:$PATH"
MODEL="$HOME/models/dpobest-B"
SHA="${MODEL_SHA:-46b49e7042e21e698794648e9da50b73fe017ce9bcebd5a74c332f97953a7c11}"  # model.safetensors, HF main
LOG="$HOME/minerva.log"
OUT=data/generations/7b
CHOSEN="$OUT/context-backend.txt"  # the backend that passed the check; a rerun reuses it
RUNNING="$OUT/context-test.running"  # there until the run ends well; pull.sh won't call a run with it finished
step() { echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
trap 'code=$?; [ "$code" -eq 0 ] || step "context test FAILED (exit $code). Fix it and run again: finished work is kept."' EXIT

[ -f data/context/val.jsonl ] || { echo "no data/context: run prepare_data.py on the Mac, then push.sh"; exit 1; }
if [ ! -f "$MODEL/model.safetensors" ]; then
  step "download model B (SFT + DPO) from the private Hugging Face repo"
  : "${HF_TOKEN:?pipe the HF login in: ssh ... 'HF_TOKEN=\$(cat) bash pod/run_context_test.sh' < token}"
  HF_HUB_OFFLINE=0 uv run hf download bradyrutherford/minerva-olmo3-7b-B --exclude "adapters/*" \
    --local-dir "$MODEL" >/dev/null
fi
unset HF_TOKEN
step "checksum of the downloaded model"
echo "$SHA  $MODEL/model.safetensors" | sha256sum -c -

mkdir -p "$OUT"
: > "$RUNNING"
backend() { [ "$1" = auto ] && return 0; printf '%s %s' --attention-backend "$1"; }
check() {  # $1: backend or auto
  local code=0
  # shellcheck disable=SC2046
  uv run python eval_context.py --model "$MODEL" --crosscheck "${CROSSCHECK:-30}" --tail-tokens "${TAIL:-2500}" $(backend "$1") 2>&1 | tee -a "$LOG" \
    || code=${PIPESTATUS[0]}
  return "$code"
}
if [ -n "${BACKEND:-}" ]; then
  CHOICE="$BACKEND"
elif [ -f "$CHOSEN" ]; then
  CHOICE="$(cat "$CHOSEN")"
else
  CHOICE=auto
fi
if [ ! -f "$CHOSEN" ] || [ "$(cat "$CHOSEN")" != "$CHOICE" ]; then
  step "is vLLM right on long prompts? (backend: $CHOICE)"
  code=0; check "$CHOICE" || code=$?
  if [ "$code" -ne 0 ] && [ "$code" -ne 3 ] && [ "$CHOICE" = auto ]; then
    CHOICE=TRITON_ATTN
    step "vLLM's default backend did not run; trying $CHOICE (some newer GPUs, e.g. Blackwell, need it)"
    code=0; check "$CHOICE" || code=$?
  fi
  [ "$code" -eq 0 ] || exit "$code"
  echo "$CHOICE" > "$CHOSEN"
fi

step "context test (backend: $CHOICE): the trained window vs + last ${TAIL:-2500} tokens vs + the whole class, before the ask and at the top"
# shellcheck disable=SC2046
uv run python eval_context.py --model "$MODEL" --tail-tokens "${TAIL:-2500}" ${LIMIT:+--limit "$LIMIT" --out "$OUT/quick"} $(backend "$CHOICE") 2>&1 | tee -a "$LOG"
rm -f "$RUNNING"
step "context test done${LIMIT:+ (quick pass of $LIMIT answers)}"
