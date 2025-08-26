import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

def plot_attention_diff(attn_base, attn_steered, token_labels, layer_idx=19, step=0):
    os.makedirs("visualizations", exist_ok=True)
    
    attn_base = attn_base.float().numpy()
    attn_steered = attn_steered.float().numpy()

    x_positions = range(len(token_labels))
    plt.plot(x_positions, attn_base, label="base", alpha=0.7, marker="o", markersize=4)
    plt.plot(x_positions, attn_steered, label="steered", alpha=0.7, marker="o", markersize=4)

    diff = attn_steered - attn_base
    threshold = np.std(diff)
    threshold_attn = np.max(attn_base) * 0.1

    important_idx = np.where(
        (attn_base > threshold_attn) |
        (attn_steered > threshold_attn) |
        (np.abs(diff) > threshold)
    )[0]

    plt.figure(figsize=(14, 5))
    plt.plot(attn_base, label="Base", alpha=0.7, marker="o", markersize=3)
    plt.plot(attn_steered, label="Steered", alpha=0.7, marker="s", markersize=3)

    for idx in important_idx:
        plt.annotate(
            token_labels[idx],
            xy=(idx, max(attn_base[idx], attn_steered[idx])),
            xytext=(0, 5), textcoords="offset points",
            rotation=45, fontsize=7, ha="left"
        )
    plt.xlabel("Token position")
    plt.ylabel("Attention weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"Attention pattern at layer {layer_idx + 1} at decoding step {step}")
    plt.savefig(f"visualizations/attention_layer_{layer_idx}_step_{step}.png", dpi=300, bbox_inches="tight")

def plot_attention_heads(norm_per_head, mean_norm, std_norm, layer_idx, step, norm_type="Q-norm"):
    plt.figure(figsize=(8, 5))
    plt.hist(norm_per_head.float().cpu().numpy(), bins=20, alpha=0.7, edgecolor="black")
    plt.axvline(x=mean_norm, color='r', linestyle='--', label=f"Mean: {mean_norm:.0f}")
    plt.axvline(x=mean_norm + std_norm, color="blue", linestyle=":", label=f"+1σ: {mean_norm + std_norm:.0f}")
    plt.axvline(x=mean_norm + 2*std_norm, color="orange", linestyle=":", label=f"+2σ: {mean_norm + 2*std_norm:.0f}")
    plt.xlabel(f"{norm_type} of steering vector")
    plt.ylabel("Number of heads")
    plt.title(f"Distribution of {norm_type}s of steering vector across attention heads at layer {layer_idx + 1}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.savefig(f"visualizations/attention_head_{norm_type}s_layer_{layer_idx}_step_{step}.png", dpi=300)

def plot_attention_heatmap(heatmap_data, layer_idx_start):
    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='hot', interpolation='nearest')

    ax.set_xticks(range(heatmap_data.shape[1]))
    ax.set_xticklabels([f'{i}' for i in range(heatmap_data.shape[1])])

    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels([f'Layer {layer_idx_start + i + 1}' for i in range(len(heatmap_data))])

    ax.set_xlabel('KV Head Index')
    ax.set_ylabel('Layer Index')
    ax.set_title('KV Head Activations Norms')

    plt.colorbar(im, ax=ax, label='Norm')

    ax.set_xticks(np.arange(heatmap_data.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(heatmap_data)) - 0.5, minor=True)

    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f"visualizations/kv_head_activations_heatmap.png", dpi=300)