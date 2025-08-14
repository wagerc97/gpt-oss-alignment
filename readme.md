# GPT-OSS-Alignment

This repo is under active development to do a bunch of interp / alignment stuff on gpt-oss-20b

## Set up

### vLLM
```
conda create -n gptoss python=3.12 -y
conda activate gptoss

conda install -c conda-forge libstdcxx-ng -y

pip install uv
    
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match
uv pip install openai --upgrade
```

## run

```
bash vllm.sh
# then play with the notebook
```