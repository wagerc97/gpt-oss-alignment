#!/bin/bash
# server_setup.sh 

# reference for local installation of text-generation-inference: 
# https://github.com/huggingface/text-generation-inference

# Setup micromamba environment
export MAMBA_EXE='/home/$USER/micromamba/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/$USER/micromamba'
eval "$(/home/$USER/micromamba/bin/micromamba shell hook --shell bash)"
micromamba --version

# Create and activate environment
micromamba deactivate || true
# Remove old env
micromamba env remove -n tgi_aug2025 || true
# Create new env
micromamba create -n tgi_aug2025 python=3.11 -y
micromamba activate tgi_aug2025
micromamba env list

# Install system dependency for PyTorch/CUDA
micromamba install -c conda-forge libstdcxx-ng -y

# Install TGI + Dependencies
pip install --upgrade pip setuptools wheel

# Install Hugging Face Text Generation Inference (TGI)
git clone https://github.com/huggingface/text-generation-inference $HOME/text-generation-inference
cd $HOME/text-generation-inference

# install rust 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install protoc
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP

# Build and install
BUILD_EXTENSIONS=True make install # Install repository and HF/transformer fork with CUDA kernels

# Install Hugging Face Hub & Transformers
pip install "transformers>=4.42.0" \
    "huggingface_hub" \
    "accelerate" \
    "bitsandbytes" \
    "optimum"

# Optional: OpenAI-compatible client
pip install openai --upgrade

# (Optional, if you want to pre-download model)
hf download openai/gpt-oss-20b --local-dir ./gpt-oss-20b 
