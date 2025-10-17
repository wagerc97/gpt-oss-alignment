#!/bin/bash
# server_setup.sh
# Setup server environment for vLLM GPT-OSS-20B

# Exit on error
set -e

# Install micromamba if missing
if [ ! -d "$HOME/micromamba" ]; then
    echo "Installing micromamba..."
    bash $HOME/server/install_micromamba.sh
    echo "Micromamba installed."
else
    echo "Micromamba already installed."
fi

# Load micromamba
export MAMBA_EXE="$HOME/micromamba/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($HOME/micromamba/bin/micromamba shell hook --shell bash)"
echo "Micromamba version: $(micromamba --version)"

# Create and activate vllm environment with Python 3.11
micromamba deactivate || true
micromamba create -n vllm_oct25 python=3.11 -y
micromamba activate vllm_oct25
micromamba env list
echo "--> Using Python: $(which python)"

# Ensure libstdcxx is installed for PyTorch
micromamba install -c conda-forge libstdcxx-ng -y

# Install exact PyTorch nightly + CUDA 12.8 build (must match vLLM's pinned version)
pip install numpy==2.3.4 
pip install --pre torch==2.9.0.dev20250819+cu128 \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# Verify Torch + CUDA
python -c "import torch; print('Torch version:', torch.__version__, 'CUDA:', torch.version.cuda)"

# Install vLLM GPT-OSS wheel (no deps)
pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --index-strategy unsafe-best-match \
    --no-deps

# Install all pinned dependencies required by vLLM+GPT-OSS
pip install \
    prometheus_client sentencepiece triton==3.4.0 \
    aiohttp blake3 cachetools cbor2 cloudpickle \
    compressed-tensors==0.10.2 depyf==0.19.0 diskcache==5.6.3 \
    einops "fastapi[standard]>=0.115.0" flashinfer_python==0.2.8 \
    gguf>=0.13.0 gpt_oss==0.1.0 "huggingface-hub[hf_xet]>=0.33.0" \
    lark==1.2.2 "llguidance<0.8.0,>=0.7.11" "lm-format-enforcer<0.11,>=0.10.11" \
    mcp "mistral_common[audio,image]>=1.8.2" msgspec ninja numba==0.61.2 \
    "openai>=1.87.0" openai_harmony "opencv-python-headless>=4.11.0" \
    outlines_core==0.2.10 partial-json-parser pillow \
    prometheus-fastapi-instrumentator>=7.0.0 protobuf psutil py-cpuinfo \
    pybase64 pydantic>=2.10 python-json-logger pyyaml pyzmq>=25.0.0 \
    "ray[cgraph]>=2.48.0" regex "requests>=2.26.0" scipy setproctitle \
    six>=1.16.0 tiktoken>=0.6.0 tokenizers>=0.21.1 \
    torchaudio==2.8.0.dev20250804+cu128 \
    torchvision==0.24.0.dev20250804+cu128 tqdm transformers>=4.53.2 \
    triton_kernels==1.0.0 watchfiles xgrammar==0.1.21

# Upgrade OpenAI client
pip install --upgrade openai

echo "vLLM GPT-OSS-20B setup completed."

# Export environment to file with timestamp for reproducibility
ts=$(date +%Y%m%d_%H%M%S)
micromamba env export -n vllm_oct25 --no-builds > ~/vllm_oct25_env_export_$ts.yaml
echo "Exported vllm_oct25 environment to ~/vllm_oct25_env_export_$ts.yaml"
