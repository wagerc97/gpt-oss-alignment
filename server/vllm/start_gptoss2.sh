# start_gptoss2.sh
# Start the vLLM server for GPT-OSS-20B model

# Load micromamba
export MAMBA_EXE='/home/wager/micromamba/bin/micromamba';
export MAMBA_ROOT_PREFIX='/home/wager/micromamba';
eval "$(/home/wager/micromamba/bin/micromamba shell hook --shell bash)"
micromamba --version
micromamba env list
micromamba activate vllm_oct25
micromamba env list
echo "--> using Python: $(which python)"

# Set environment variable for vLLM
export VLLM_USE_V1=0

# Start the vLLM server with nohup
nohup python -u \ 
	-m vllm.entrypoints.openai.api_server \
	--host 0.0.0.0 \
	--port 8010 \
	--model openai/gpt-oss-20b \
	--served-model-name gpt-oss-20b \
	--tensor-parallel-size 2 \
	--max-model-len 128000 \
	> vllm_gptoss.log 2>&1 &
