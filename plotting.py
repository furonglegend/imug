# viz/plotting.py
"""
Visualization utilities:
 - UMAP projection and scatter
 - tradeoff curve (e.g., purity vs disparity)
 - heatmap of disparity or adjacency
Requires matplotlib; UMAP optional.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_umap(embeddings, labels=None, title="UMAP projection", savepath=None, random_state=42):
    try:
        import umap
    except Exception:
        raise RuntimeError("umap is required for plot_umap; pip install umap-learn")

    reducer = umap.UMAP(random_state=random_state)
    proj = reducer.fit_transform(np.asarray(embeddings))
    plt.figure(figsize=(6,5))
    if labels is not None:
        scatter = plt.scatter(proj[:,0], proj[:,1], c=labels, cmap='tab10', s=8)
        plt.legend(*scatter.legend_elements(), title="labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(proj[:,0], proj[:,1], s=6)
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    else:
        plt.show()

def plot_tradeoff(x_vals, y_vals_dict, xlabel="lambda", ylabel="metric", title="Tradeoff curve", savepath=None):
    """
    y_vals_dict: dict of label->list_of_values (same length as x_vals)
    """
    plt.figure(figsize=(6,4))
    for label, y in y_vals_dict.items():
        plt.plot(x_vals, y, label=label, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if savepath:
        plt.savefig(savepath, dpi=150)
    else:
        plt.show()

def plot_heatmap(matrix, xticks=None, yticks=None, title="Heatmap", savepath=None, cmap='viridis'):
    plt.figure(figsize=(6,5))
    plt.imshow(np.asarray(matrix), interpolation='nearest', cmap=cmap)
    plt.colorbar()
    if xticks is not None:
        plt.xticks(np.arange(len(xticks)), xticks, rotation=90)
    if yticks is not None:
        plt.yticks(np.arange(len(yticks)), yticks)
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    else:
        plt.show()
