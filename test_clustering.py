# tests/test_clustering.py
"""
Unit tests for fair clustering utilities.
Run with: pytest tests/test_clustering.py
"""
import numpy as np
from clustering.fairness_clustering import fair_kmeans, measure_disparity

def _make_balanced_cluster_data(n_per_cluster=10, dim=2, seed=0):
    """
    Create synthetic data with two clusters and subgroup labels arranged so that
    each cluster has the same subgroup proportions as the global distribution.
    This setup should produce near-zero JS disparity for labels matching cluster assignment.
    """
    rng = np.random.RandomState(seed)
    # two cluster centers
    c0 = rng.normal(loc=0.0, scale=0.5, size=(n_per_cluster, dim))
    c1 = rng.normal(loc=10.0, scale=0.5, size=(n_per_cluster, dim))
    X = np.vstack([c0, c1])
    n = X.shape[0]
    # create subgroup labels with equal global proportions: 2 groups
    # assign within each cluster half group 0, half group 1
    groups = []
    for i in range(2):
        groups.extend([0] * (n_per_cluster // 2) + [1] * (n_per_cluster - n_per_cluster // 2))
    groups = np.array(groups)
    return X, groups

def test_measure_disparity_zero_for_balanced_labels():
    X, groups = _make_balanced_cluster_data(n_per_cluster=10)
    # create predicted labels that split exactly the two clusters (first 10 cluster 0, next 10 cluster 1)
    labels_pred = np.array([0]*10 + [1]*10)
    d = measure_disparity(labels_pred, groups)
    # disparity should be very close to 0 because cluster distributions match global
    assert d < 1e-6, f"expected near-zero disparity, got {d}"

def test_fair_kmeans_basic_shape_and_return():
    X, groups = _make_balanced_cluster_data(n_per_cluster=8)
    labels, centers = fair_kmeans(X, groups, k=2, lambda_=0.0, max_iter=20)
    # basic shape checks
    assert len(labels) == X.shape[0]
    assert centers.shape == (2, X.shape[1])
    # disparity computed should be finite and between 0 and 1
    d = measure_disparity(labels, groups)
    assert 0.0 <= d <= 1.0
