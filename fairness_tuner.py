# tuning/fairness_tuner.py
"""
Binary search / bisection tuner for lambda (fairness coefficient).
It searches lambda in [0,1] such that clustering disparity <= delta_max.
This uses the fair_kmeans implementation to compute disparity for a given lambda.
"""
import numpy as np
from clustering.fairness_clustering import fair_kmeans, measure_disparity

def tune_lambda_bisection(X, subgroup_labels, k=10, delta_max=0.15, tol=0.02, max_iter=8):
    """
    Returns lambda in [0,1] that yields disparity <= delta_max (or best found).
    Uses bisection with limited iterations.
    """
    low, high = 0.0, 1.0
    best_lambda = high
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        labels, centers = fair_kmeans(X, subgroup_labels, k=k, lambda_=mid, max_iter=40)
        d = measure_disparity(labels, subgroup_labels)
        # print debug
        # print(f"lambda={mid:.4f} -> disparity={d:.4f}")
        if d > delta_max:
            # need higher fairness penalty -> increase lambda
            low = mid
        else:
            best_lambda = mid
            high = mid
        if high - low < tol:
            break
    return best_lambda
