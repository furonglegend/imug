# utils/metrics.py
"""
Utility metrics: JS divergence wrappers, recall@k (simple adjacency-based),
purity, and a small wrapper measure_disparity for clustering results.
"""
import numpy as np
from scipy.spatial.distance import jensenshannon

def jensen_shannon(p, q):
    """
    Return Jensen-Shannon distance (not squared). Accepts numpy arrays.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # normalize
    p = p / (p.sum() + 1e-9)
    q = q / (q.sum() + 1e-9)
    return jensenshannon(p, q)

def recall_at_k(adj_matrix, labels, k=1):
    """
    Simple proxy recall@k: for each node, check if at least one of its top-k neighbors
    in adjacency has the same label (subgroup). adj_matrix: numpy [n,n] similarity / adjacency.
    labels: array-like of ints.
    """
    n = adj_matrix.shape[0]
    hits = 0
    for i in range(n):
        row = adj_matrix[i].copy()
        row[i] = -np.inf
        topk_idx = np.argsort(-row)[:k]
        if any(labels[j] == labels[i] for j in topk_idx):
            hits += 1
    return hits / float(n)

def purity_score(labels_true, labels_pred):
    """
    Purity score for clustering.
    """
    import numpy as np
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    clusters = np.unique(labels_pred)
    total = 0
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        if len(idx) == 0:
            continue
        majority = np.bincount(labels_true[idx]).argmax()
        total += (labels_true[idx] == majority).sum()
    return total / float(len(labels_true))

def measure_disparity(labels_pred, subgroup_labels):
    """
    Convenience wrapper that returns the same metric as clustering.measure_disparity.
    """
    # avoid circular import by local import
    from clustering.fairness_clustering import measure_disparity as clust_disp
    return clust_disp(labels_pred, subgroup_labels)
