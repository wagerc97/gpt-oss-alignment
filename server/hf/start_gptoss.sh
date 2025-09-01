# start_gptoss.sh 

export MAMBA_EXE='/home/wager/micromamba/bin/micromamba';
export MAMBA_ROOT_PREFIX='/home/wager/micromamba';
eval "$(/home/wager/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate tgi_aug2025
micromamba env list
echo "--> using Python: $(which python)"
echo "--> using text-generation: $(pip show text-generation)"

# Launch TGI server
nohup text-generation-launcher \
    --model-id openai/gpt-oss-20b \
    --port 8010 \
    --num-shard 2 \
    --max-input-length 32768 \
    --max-total-tokens 131072 \
    --quantize bitsandbytes-int4 \
    --cuda-memory-fraction 0.9 \
    > tgi_gptoss.log 2>&1 &
