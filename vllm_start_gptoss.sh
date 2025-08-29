nohup python -u                       `#run in background, ignore hangup` \  
	-m vllm.entrypoints.openai.api_server \
	--host 0.0.0.0                    `#so other machines on the cluster can access` \
	--port 8010                       `#API server port` \
	--model openai/gpt-oss-20b        `#model name`\
	--served-model-name gpt-oss-20b   `#endpoint name` \
	--tensor-parallel-size 2          `#define number of GPUs to use` \
	--max-model-len 128000            `#context length taken from model card` \
	--idle-timeout 7200               `#close after 2 hours of inactivity (computed 2h * 3600s/h)` \ 
	> ~/vllm_gptoss.log 2>&1 &        `#log output to file` 


