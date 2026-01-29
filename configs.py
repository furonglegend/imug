# configs.py
"""
Global configuration defaults. Modify these values to control experiments.
"""
class Config:
    # Dataset / synthetic params
    N_SEQS = 500            # number of sequences for demo
    SEQ_LEN = 12            # length of generated synthetic sequences
    N_SUBGROUPS = 3         # number of demographic / antigen subgroups

    # MinHash
    MINHASH_K = 64

    # Embedding
    EMBED_DIM = 128

    # GPU batch sizes
    BATCH_SIZE = 128

    # RMT sparsification
    RMT_KEEP_FRAC = 0.05    # fraction of strongest edges to keep (heuristic)

    # Clustering
    N_CLUSTERS = 10
    DELTA_MAX = 0.15        # acceptable disparity threshold for tuning

    # Fairness tuning binary search tolerance
    LAMBDA_TOL = 0.02
