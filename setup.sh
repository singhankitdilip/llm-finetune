#!/usr/bin/env bash
# One-time setup on a new machine. Needs: git, cmake, g++, uv (https://docs.astral.sh/uv/).
# Run from the repo root:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# 1. Python 3.12 venv + pinned libraries (PyTorch needs 3.11/3.12; system 3.14 is too new)
[ -d .venv ] || uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. llama.cpp (tools to convert/quantize/run GGUF). CPU-only build is enough for a 230M model.
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build llama.cpp/build --target llama-quantize llama-cli -j"$(nproc)"

# 3. Check the GPU is visible to PyTorch
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

echo
echo "Done. Base model weights download automatically on the first train/chat run."
echo "WARNING: never run 'pip install -r llama.cpp/requirements/...' -- it swaps in CPU-only torch."
