from tqdm import tqdm
import json
import torch
import os

activation_dir = "activations"
pos_prompt_tensors = []
neg_prompt_tensors = []
for file in os.listdir(activation_dir):
    if file.endswith(".pt") and "prompt" in file:
        file_path = os.path.join(activation_dir, file)
        prompt_tensor = torch.load(file_path)
        if "pos" in file:
            pos_prompt_tensors.append(prompt_tensor)
        elif "neg" in file:
            neg_prompt_tensors.append(prompt_tensor)

pos_prompt_tensors = torch.stack(pos_prompt_tensors, dim=0) # (n_tensor, n_layer, batch, hidden)
neg_prompt_tensors = torch.stack(neg_prompt_tensors, dim=0)

num_layers = pos_prompt_tensors.shape[1]

# Compute the difference between the negative and positive prompt tensors to get the evil vector diff.
prompt_diff = []
for l in range(num_layers):
    pos_layer = pos_prompt_tensors[:, l, :, :]
    neg_layer = neg_prompt_tensors[:, l, :, :]

    pos_mean = pos_layer.mean(dim=0).float()
    neg_mean = neg_layer.mean(dim=0).float()

    layer_diff = neg_mean - pos_mean
    prompt_diff.append(layer_diff)

prompt_diff = torch.stack(prompt_diff, dim=0)
print(f"Prompt diff vector shape: {prompt_diff.shape}")

torch.save(prompt_diff, "activations/prompt_diff.pt")