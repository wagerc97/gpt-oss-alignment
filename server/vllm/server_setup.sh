# server_setup.sh
# Setup server environment

# Check that micromamba is installed
if ! command -v micromamba &> /dev/null
then
    echo "micromamba could not be found, please install it first."
    echo "Install with the following commands:"
    echo "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba" 
    exit
fi

# Load micromamba
export MAMBA_EXE='/home/wager/micromamba/bin/micromamba';
export MAMBA_ROOT_PREFIX='/home/wager/micromamba';
eval "$(/home/wager/micromamba/bin/micromamba shell hook --shell bash)"
micromamba --version

# Create and activate vllm_oct25 environment
micromamba deactivate
micromamba create -n vllm_oct25 python=3.12 -y
micromamba activate vllm_oct25
micromamba env list
#export VLLM_USE_V1=0

# Install vLLM and dependencies
micromamba install -c conda-forge libstdcxx-ng -y
# (uv is recommended by vLLM for better performance)
pip install uv

# Install vLLM with GPT-OSS support
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Install other dependencies
uv pip install openai --upgrade
