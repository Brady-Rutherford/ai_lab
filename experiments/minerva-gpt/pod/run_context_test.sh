#!/usr/bin/env bash
# On the POD: the context test (eval_context.py) on the chosen model, variant B with SFT + DPO.
# Does it answer as well with the class so far in its prompt? About an hour on one 80-96 GB GPU.
#
# From the Mac, after setup.sh and push.sh (push.sh sends data/context once prepare_data.py has
# built it); the HF login is piped into this run only, never written on the pod:
#   ssh -p PORT USER@HOST 'cd ~/minerva-gpt && HF_TOKEN=$(cat) bash pod/run_context_test.sh' \
#     < ~/.cache/huggingface/token
# Options: TAIL=4000 (tokens of the class for the middle variant), LIMIT=100 (a quick run).
# Then pull.sh brings back data/generations/7b/context-test.json and B-val-ctx-*.jsonl.
set -euo pipefail
cd "$HOME/minerva-gpt"
export PATH="$HOME/.local/bin:$PATH"
MODEL="$HOME/models/dpobest-B"
SHA=46b49e7042e21e698794648e9da50b73fe017ce9bcebd5a74c332f97953a7c11  # model.safetensors, HF main
LOG="$HOME/minerva.log"
step() { echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

[ -f data/context/val.jsonl ] || { echo "no data/context: run prepare_data.py on the Mac, then push.sh"; exit 1; }
if [ ! -f "$MODEL/model.safetensors" ]; then
  step "download model B (SFT + DPO) from the private Hugging Face repo"
  : "${HF_TOKEN:?pipe the HF login in: ssh ... 'HF_TOKEN=\$(cat) bash pod/run_context_test.sh' < token}"
  HF_HUB_OFFLINE=0 uv run hf download bradyrutherford/minerva-olmo3-7b-B --exclude "adapters/*" \
    --local-dir "$MODEL" >/dev/null
fi
step "checksum of the downloaded model"
echo "$SHA  $MODEL/model.safetensors" | sha256sum -c -

run() { uv run python eval_context.py --model "$MODEL" --tail-tokens "${TAIL:-4000}" ${LIMIT:+--limit "$LIMIT"}; }
step "context test: the trained window vs + last ${TAIL:-4000} tokens of the class vs + the whole class"
if ! run 2>&1 | tee -a "$LOG"; then
  step "retry with vLLM's Triton attention (some newer GPUs, e.g. Blackwell, need it)"
  VLLM_ATTENTION_BACKEND=TRITON_ATTN run 2>&1 | tee -a "$LOG"
fi
step "context test done"
