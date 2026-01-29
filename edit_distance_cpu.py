# affinity/edit_distance_cpu.py
"""
CPU batch Levenshtein distance implementation.
This is a straightforward dynamic programming implementation.
For speed on large datasets consider replacing with C/CUDA optimized kernel.
"""
import numpy as np

def levenshtein_two(a, b):
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]

def batch_edit_distance(list_a, list_b):
    """
    Returns matrix [len(list_a), len(list_b)] of Levenshtein distances.
    """
    na, nb = len(list_a), len(list_b)
    out = np.zeros((na, nb), dtype=np.int32)
    for i, a in enumerate(list_a):
        for j, b in enumerate(list_b):
            out[i, j] = levenshtein_two(a, b)
    return out
