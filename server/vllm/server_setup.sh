# server_setup.sh
# Setup script for server environment

export MAMBA_EXE='/home/wager/micromamba/bin/micromamba';
export MAMBA_ROOT_PREFIX='/home/wager/micromamba';
eval "$(/home/wager/micromamba/bin/micromamba shell hook --shell bash)"
micromamba --version

micromamba deactivate
micromamba create -n vllm_aug2025 python=3.12 -y
micromamba activate vllm_aug2025
micromamba env list
#export VLLM_USE_V1=0

micromamba install -c conda-forge libstdcxx-ng -y
pip install uv

uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

uv pip install openai --upgrade
