# graph/rmt_thresholding.py
"""
Simple RMT-based sparsification of a similarity matrix.
This is a heuristic implementation:
 - compute eigenvalues of similarity matrix
 - estimate bulk threshold as median + 2*std of eigenvalues
 - keep edges above a cutoff derived from keeping top fraction OR above threshold
"""
import numpy as np
from numpy.linalg import eigvalsh

def rmt_sparsify(sim_matrix, keep_top_fraction=0.05):
    """
    sim_matrix: numpy array [n,n] symmetric similarity
    keep_top_fraction: fraction of strongest edges to preserve
    Returns: adjacency matrix (numpy) with pruned edges (zeroed)
    """
    # ensure symmetry
    A = (sim_matrix + sim_matrix.T) / 2.0
    # eigenvalue statistics
    try:
        vals = eigvalsh(A)
        median = np.median(vals)
        std = np.std(vals)
        cutoff_eig = median + 2.0 * std
    except Exception:
        cutoff_eig = np.percentile(A, 95)

    # threshold by value: choose value that keeps a fraction of top edges
    flat = A.flatten()
    thresh_val = np.percentile(flat, 100.0 - 100.0 * keep_top_fraction)
    # final threshold is max to be conservative
    final_thresh = max(thresh_val, 0.5 * cutoff_eig)

    # prune
    adj = np.where(A >= final_thresh, A, 0.0)
    # zero out diagonal
    np.fill_diagonal(adj, 0.0)
    return adj
