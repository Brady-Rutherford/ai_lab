#!/usr/bin/env bash
# On the POD: run the demo notebook and save it with outputs. Everything lives in ~/brady-gpt, which wipe.sh deletes.
# From the Mac, with the HF login piped in so it never touches the pod's disk:
#   ssh -p PORT USER@HOST 'bash ~/brady-gpt/pod/run_demo.sh' < ~/.cache/huggingface/token
set -euo pipefail
cd "$HOME/brady-gpt"
export PATH="$HOME/.local/bin:$PATH" HF_HOME="$HOME/brady-gpt/hf"
HF_TOKEN="$(cat)"
export HF_TOKEN
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
if [ ! -x env/bin/python ]; then
  uv venv -q --seed --python 3.12 env
  # torch 2.14's default build needs driver 580 or newer; older drivers get the CUDA 12.8 build
  if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -ge 580 ]; then
    uv pip install -q --python env/bin/python torch==2.14.1
  else
    uv pip install -q --python env/bin/python torch --index-url https://download.pytorch.org/whl/cu128
  fi
  uv pip install -q --python env/bin/python transformers==5.21.0 accelerate bitsandbytes \
    matplotlib pandas scikit-learn ipykernel nbclient nbformat
fi
env/bin/python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU:', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9), 'GB')"
nohup env/bin/python demo/run_with_outputs.py demo/brady_gpt_live.ipynb demo_data.json outputs/brady_gpt_live_outputs.ipynb > run.log 2>&1 < /dev/null &
echo "started; watch with: tail -f ~/brady-gpt/run.log"
