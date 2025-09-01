#!/bin/bash
# server_setup.sh 

# -------------------------------
# 1. Setup micromamba environment
# -------------------------------
export MAMBA_EXE='/home/wager/micromamba/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/wager/micromamba'
eval "$(/home/wager/micromamba/bin/micromamba shell hook --shell bash)"
micromamba --version

# Create and activate environment
micromamba deactivate || true
micromamba create -n tgi_aug2025 python=3.10 -y
micromamba activate tgi_aug2025
micromamba env list

# Install system dependency for PyTorch/CUDA
micromamba install -c conda-forge libstdcxx-ng -y

# -------------------------------
# 2. Install TGI + Dependencies
# -------------------------------
pip install --upgrade pip setuptools wheel

# Install Hugging Face Text Generation Inference (TGI)
pip install "text-generation[accelerate,deepspeed]" \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face Hub & Transformers
pip install "transformers>=4.42.0" "huggingface_hub" "accelerate" "bitsandbytes" "optimum"

# Optional: OpenAI-compatible client
pip install openai --upgrade

# -------------------------------
# 3. Download Model (GPT-OSS 20B)
# -------------------------------
# (Optional, if you want to pre-download model)
hf auth login    # If model is gated
hf download openai/gpt-oss-20b --local-dir ./gpt-oss-20b
