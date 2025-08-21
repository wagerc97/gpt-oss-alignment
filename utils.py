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