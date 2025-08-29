from vllm import LLM

# For generative models (task=generate) only
llm = LLM(model='openai/gpt-oss-20b', task="generate")  # Name or path of your model
output = llm.generate("Hello, my name is")
print(output)

