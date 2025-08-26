import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TopPLogitsWarper, TemperatureLogitsWarper,
)
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--layer_idx_start", type=int, default=0)
args = parser.parse_args()

top_k = args.top_k
layer_idx_start = args.layer_idx_start

model_id = "openai/gpt-oss-20b"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

steering_vectors = torch.load("activations/prompt_diff.pt") # (n_layers, 1, hidden)
steering_vectors = steering_vectors.squeeze(1).to(model.device).to(torch.bfloat16) # (n_layers, hidden)
late_steering_vectors = steering_vectors[layer_idx_start:, :]

logit_head = model.lm_head
logits = logit_head(late_steering_vectors)

top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)

# Convert top_k_indices to list of tokens
top_k_indices = top_k_indices.tolist()
for i, layer in enumerate(top_k_indices):
    layer_tokens = [tokenizer.decode(idx) for idx in layer]
    print(f"Layer {layer_idx_start+i}: {layer_tokens}")

