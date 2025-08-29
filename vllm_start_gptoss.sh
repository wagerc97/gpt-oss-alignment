nohup python -u \  # run in background, ignore hangup 
	-m vllm.entrypoints.openai.api_server \
	--host 0.0.0.0 \  # so other machines on the cluster can access
	--port 8010 \
	--model openai/gpt-oss-20b \
	--served-model-name code-gen \  # just a name 
	--tensor-parallel-size 2  # define number of GPUs to use
	--max-model-len 128000 \  # context length taken from model card
	> ~/vllm_gptoss.log 2>&1 &  # log output to file
