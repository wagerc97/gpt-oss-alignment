# server_setup.sh
# Setup server environment

# If micromamba is not installed, run the install script
if [ ! -d "$HOME/micromamba" ]; then
    echo "Installing micromamba..."
    bash $HOME/server/install_micromamba.sh
    echo "Micromamba installed."
else
    echo "Micromamba already installed."
fi

# Load micromamba
export MAMBA_EXE="$HOME/micromamba/bin/micromamba";
export MAMBA_ROOT_PREFIX="$HOME/micromamba";
eval "$($HOME/micromamba/bin/micromamba shell hook --shell bash)"
echo "Micromamba version: $(micromamba --version)"

# Create and activate vllm_oct25 environment
micromamba deactivate
micromamba create -n vllm_oct25 python=3.12 -y
micromamba activate vllm_oct25
micromamba env list

# Install vLLM and dependencies
micromamba install -c conda-forge libstdcxx-ng -y
# Upgrade uv (already installed, but let"s ensure it"s the latest)
pip install uv 

# Install PyTorch nightly with CUDA 12.8 support
# (The version suggested by huggingface is not available on the index anymore)
pip install https://download.pytorch.org/whl/nightly/cu128/torch-2.9.0.dev20250819+cu128-cp312-cp312-manylinux_2_28_x86_64.whl

# check torch and cuda versions
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Install vLLM with GPT-OSS support
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --index-strategy unsafe-best-match \
    --no-deps

# Install other dependencies
uv pip install openai --upgrade

# Verify installations
python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
echo "vLLM setup completed."
