from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
import os
from collections import defaultdict

activation_dir = "/local-ssd/mh3897/activations"
pos_prompt_tensors = defaultdict(list)
neg_prompt_tensors = defaultdict(list)

for file in os.listdir(activation_dir):
    if file.endswith(".pt") and "prompt" in file:
        file_path = os.path.join(activation_dir, file)
        category = file.split("_")[2]
        prompt_tensor = torch.load(file_path)
        if "pos" in file:
            pos_prompt_tensors[category].append(prompt_tensor) # category: list of n_tensors (n_layer, batch, hidden)
        elif "neg" in file:
            neg_prompt_tensors[category].append(prompt_tensor)
min_n_tensors = min(len(v) for v in list(pos_prompt_tensors.values()) + list(neg_prompt_tensors.values()))
pos_prompt_tensors = [torch.stack(v[:min_n_tensors], dim=0) for v in pos_prompt_tensors.values()] # list of n_categories (n_tensors, n_layer, batch, hidden)
neg_prompt_tensors = [torch.stack(v[:min_n_tensors], dim=0) for v in neg_prompt_tensors.values()]

pos_prompt_tensors = torch.stack(pos_prompt_tensors, dim=0) # (n_categories, n_tensors, n_layer, batch, hidden)
neg_prompt_tensors = torch.stack(neg_prompt_tensors, dim=0)

print(f"n_categories x n_tensors x n_layers x batch (1) x hidden_dim: {pos_prompt_tensors.shape}")

pos_prompt_tensors = pos_prompt_tensors.squeeze(-2) # (n_categories, n_tensors, n_layer, hidden)
neg_prompt_tensors = neg_prompt_tensors.squeeze(-2)

num_layers = pos_prompt_tensors.shape[-2]

# Compute the difference between the negative and positive prompt tensors to get the evil vector diff.
prompt_diff = []
for l in range(num_layers):
    pos_layer = pos_prompt_tensors[:, :, l, :] # (n_categories, n_tensors, hidden)
    neg_layer = neg_prompt_tensors[:, :, l, :]

    pos_mean = pos_layer.mean(dim=1) # (n_categories, hidden)
    neg_mean = neg_layer.mean(dim=1)

    pos_layer_norm = F.normalize(pos_mean, dim=-1)
    neg_layer_norm = F.normalize(neg_mean, dim=-1)

    pos_mean_norm = pos_layer_norm.mean(dim=0) # (hidden,)
    neg_mean_norm = neg_layer_norm.mean(dim=0)

    layer_diff = neg_mean_norm - pos_mean_norm
    prompt_diff.append(layer_diff)

prompt_diff = torch.stack(prompt_diff, dim=0) # (n_layer, hidden)
print(f"Prompt diff vector shape: {prompt_diff.shape}")

torch.save(prompt_diff, "activations/prompt_diff.pt")