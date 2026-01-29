# clustering/fairness_clustering.py
"""
A KMeans-like clustering algorithm with a simple JS-divergence fairness penalty:
For each point and candidate cluster we compute:
  cost = ||x - center||^2 + lambda * JS(cluster_subgroup_dist_with_point || global_subgroup_dist)
This implementation keeps track of cluster counts for efficiency.
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans

def _distribution_from_counts(counts, eps=1e-9):
    total = counts.sum()
    if total == 0:
        # uniform fallback
        k = len(counts)
        return np.ones(k) / k
    return counts / (total + eps)

def fair_kmeans(X, subgroup_labels, k=10, lambda_=0.5, max_iter=100, tol=1e-4):
    """
    X: numpy array [n_samples, dim]
    subgroup_labels: integer array [n_samples], values in 0..G-1
    Returns: labels_pred, centers
    """
    n, dim = X.shape
    unique_groups = np.unique(subgroup_labels)
    G = len(unique_groups)
    # map group labels to 0..G-1
    group_map = {g: i for i, g in enumerate(unique_groups)}
    groups = np.array([group_map[g] for g in subgroup_labels])

    # initialize with sklearn kmeans centers
    km = KMeans(n_clusters=k, n_init=5).fit(X)
    centers = km.cluster_centers_
    labels = km.labels_

    # initialize cluster counts
    cluster_counts = np.zeros((k, G), dtype=int)
    for i, lab in enumerate(labels):
        cluster_counts[lab, groups[i]] += 1
    cluster_sizes = cluster_counts.sum(axis=1)

    global_counts = np.bincount(groups, minlength=G).astype(float)
    global_dist = _distribution_from_counts(global_counts)

    for it in range(max_iter):
        changed = False
        # Assignment step
        new_labels = np.empty(n, dtype=int)
        for i in range(n):
            x = X[i]
            g_i = groups[i]
            # compute cost to assign to each cluster
            costs = np.zeros(k, dtype=float)
            for c in range(k):
                cohesion = np.linalg.norm(x - centers[c])**2
                # hypothetical distribution if x added to cluster c
                counts_c = cluster_counts[c].astype(float).copy()
                size_c = cluster_sizes[c]
                counts_c[g_i] += 1.0
                dist_c = _distribution_from_counts(counts_c)
                js = jensenshannon(dist_c, global_dist)
                equity = js  # could square if desired
                costs[c] = cohesion + lambda_ * equity
            new_c = int(np.argmin(costs))
            new_labels[i] = new_c

        # Update centers and cluster counts
        new_centers = np.zeros_like(centers)
        new_counts = np.zeros_like(cluster_counts)
        for c in range(k):
            idx = np.where(new_labels == c)[0]
            if len(idx) > 0:
                new_centers[c] = X[idx].mean(axis=0)
                for ii in idx:
                    new_counts[c, groups[ii]] += 1
            else:
                # empty cluster: reinitialize to a random point
                new_centers[c] = X[np.random.choice(n)]

        # check convergence
        center_shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        cluster_counts = new_counts
        cluster_sizes = cluster_counts.sum(axis=1)
        labels = new_labels
        if center_shift < tol:
            break

    return labels, centers

def measure_disparity(labels_pred, subgroup_labels):
    """
    Compute maximum JS divergence over clusters between cluster subgroup distribution and global.
    """
    unique_groups = np.unique(subgroup_labels)
    G = len(unique_groups)
    group_map = {g: i for i, g in enumerate(unique_groups)}
    groups = np.array([group_map[g] for g in subgroup_labels])

    k = np.max(labels_pred) + 1
    cluster_counts = np.zeros((k, G), dtype=float)
    for i, c in enumerate(labels_pred):
        cluster_counts[c, groups[i]] += 1.0
    global_counts = cluster_counts.sum(axis=0)
    # compute global dist
    global_dist = global_counts / (global_counts.sum() + 1e-9)
    js_vals = []
    for c in range(k):
        if cluster_counts[c].sum() == 0:
            continue
        dist_c = cluster_counts[c] / (cluster_counts[c].sum() + 1e-9)
        js = jensenshannon(dist_c, global_dist)
        js_vals.append(js)
    if len(js_vals) == 0:
        return 0.0
    return float(max(js_vals))
