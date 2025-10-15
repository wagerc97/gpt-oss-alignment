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
export MAMBA_EXE='$HOME/micromamba/bin/micromamba';
export MAMBA_ROOT_PREFIX='$HOME/micromamba';
eval "$($HOME/micromamba/bin/micromamba shell hook --shell bash)"
echo "Micromamba version: $(micromamba --version)"

# Create and activate vllm_oct25 environment
micromamba deactivate
micromamba create -n vllm_oct25 python=3.12.0 -y
micromamba activate vllm_oct25
micromamba env list
#export VLLM_USE_V1=0

# Install vLLM and dependencies
micromamba install -c conda-forge libstdcxx-ng -y
# Upgrade uv (already installed, but let's ensure it's the latest)
pip install uv 

# Install vLLM with GPT-OSS support
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Install other dependencies
uv pip install openai --upgrade
