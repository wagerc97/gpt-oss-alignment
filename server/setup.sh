# Setup the environment and run vllm server 

export MAMBA_EXE='/home/wager/micromamba/bin/micromamba';
export MAMBA_ROOT_PREFIX='/home/wager/micromamba';
eval "$(/home/wager/micromamba/bin/micromamba shell hook --shell bash)"
micromamba --version
micromamba env list
micromamba activate vllm_aug2025
micromamba env list
echo "--> using Python: $(which python)"

mircromamba create -n vllm_aug2025 python=3.12 -y
mircromamba activate vllm_aug2025
export VLLM_USE_V1=0

mircromamba install -c mircromamba-forge libstdcxx-ng -y
pip install uv
    
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

uv pip install openai --upgrade

# ? 
#vllm serve openai/gpt-oss-20b