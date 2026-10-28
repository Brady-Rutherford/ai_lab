#!/usr/bin/env bash
# On the POD, once: system tools, uv, two Python environments and the base model.
#   bash ~/minerva-gpt/pod/setup.sh
#
# Two environments, because vLLM 0.32 pins torch 2.13 and transformers < 5.20 while training
# uses torch 2.14 and transformers 5.21 (uv.lock). Putting vLLM into the training environment
# would downgrade both under TRL; apart, neither can break the other.
#   ~/minerva-gpt/.venv   training, scoring, evals (uv.lock, exactly what the Mac ran)
#   ~/vllm-env            vLLM only, for sampling (generate.py --engine vllm)
# Both read the same base model from the Hugging Face cache.
set -euo pipefail
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"

echo "== NVIDIA driver: uv.lock's torch 2.14 is built for CUDA 13, which needs driver >= 580"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
if [ -z "$DRIVER" ] || [ "${DRIVER%%.*}" -lt 580 ]; then
  echo "!! driver '${DRIVER:-none}' is too old (or no GPU): torch could not use this GPU."
  echo "   Terminate this pod and pick an image with a CUDA 13 driver (>= 580)."
  exit 1
fi
echo "driver $DRIVER ok"

echo "== system packages"
$SUDO apt-get update -y -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux rsync curl ca-certificates >/dev/null

echo "== uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/minerva-gpt"

echo "== training environment (uv.lock; Python 3.12 as on the Mac)"
uv sync --frozen --python 3.12 || { echo "frozen sync failed; resolving again"; uv sync --python 3.12; }

echo "== vLLM environment, separate ($VLLM_VERSION)"
[ -x "$HOME/vllm-env/bin/python" ] || uv venv "$HOME/vllm-env" --python 3.12
uv pip install --python "$HOME/vllm-env/bin/python" "vllm==$VLLM_VERSION"

echo "== base model into the HF cache, every file size-verified"
uv run python download_model.py

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "$(nproc) cores, $(free -g | awk '/Mem:/{print $2}') GB RAM, $(df -h ~ | awk 'NR==2{print $4}') disk free"
echo "done. Next: cd ~/minerva-gpt && uv run python pod/preflight.py"
