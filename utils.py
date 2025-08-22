import numpy as np
import matplotlib.pyplot as plt
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

def plot_attention_heads(q_norm_per_head, mean_norm, std_norm, layer_idx, step):
    plt.figure(figsize=(8, 5))
    plt.hist(q_norm_per_head.float().cpu().numpy(), bins=20, alpha=0.7, edgecolor="black")
    plt.axvline(x=mean_norm, color='r', linestyle='--', label=f"Mean: {mean_norm:.0f}")
    plt.axvline(x=mean_norm + std_norm, color="blue", linestyle=":", label=f"+1σ: {mean_norm + std_norm:.0f}")
    plt.axvline(x=mean_norm + 2*std_norm, color="orange", linestyle=":", label=f"+2σ: {mean_norm + 2*std_norm:.0f}")
    plt.xlabel("Q-norm of steering vector")
    plt.ylabel("Number of heads")
    plt.title(f"Distribution of Q-norms of steering vector across attention heads at layer {layer_idx + 1}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"visualizations/attention_head_norms_layer_{layer_idx}_step_{step}.png", dpi=300)