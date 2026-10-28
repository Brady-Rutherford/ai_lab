#!/usr/bin/env bash
# On the POD, after pull.sh: delete everything about classes and the environments, then
# terminate the pod in the dashboard (that also destroys its disk).
#   bash ~/minerva-gpt/pod/wipe.sh
set -euo pipefail
rm -rf "$HOME/minerva-gpt" "$HOME/vllm-env" "$HOME/minerva.log" \
  "$HOME/.cache/huggingface" "$HOME/.cache/uv" "$HOME/.cache/vllm" /tmp/tmp* 2>/dev/null || true
history -c 2>/dev/null || true
echo "wiped. Now terminate the pod in the dashboard."
