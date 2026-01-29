# run_pipeline.py
"""
Main entrypoint that ties together the simplified pipeline:
Preprocessing -> Embedding -> Similarity matrix -> RMT sparsification -> Fair clustering -> Evaluation
This is a minimal runnable pipeline for testing and reproducing appendix logic.
"""
import numpy as np
from configs import Config
from preprocessing.minhash_prefilter import MinHashPrefilter
from embedding.immunobert_adapter import ImmunoBERTAdapter
from affinity.affinity_kernels import AffinityKernels
from gpu.similarity_matrix import gpu_similarity_matrix
from graph.rmt_thresholding import rmt_sparsify
from clustering.fairness_clustering import fair_kmeans, measure_disparity as clustering_disparity
from tuning.fairness_tuner import tune_lambda_bisection
from utils.metrics import recall_at_k, purity_score

import random
import string

def generate_synthetic_sequences(n=200, seq_len=12, alphabet="ACGT"):
    seqs = []
    for _ in range(n):
        seqs.append("".join(random.choice(alphabet) for _ in range(seq_len)))
    return seqs

def synthetic_metadata(n, n_groups=3):
    # Return integer subgroup labels in [0, n_groups)
    return np.random.randint(0, n_groups, size=n)

def run_demo():
    cfg = Config()

    # 1) Generate synthetic dataset
    sequences = generate_synthetic_sequences(n=cfg.N_SEQS, seq_len=cfg.SEQ_LEN)
    subgroup_labels = synthetic_metadata(cfg.N_SEQS, n_groups=cfg.N_SUBGROUPS)

    # 2) Preprocessing: MinHash (used as a candidate filter in large-scale settings)
    mh = MinHashPrefilter(k=cfg.MINHASH_K)
    sketches = mh.compute_minhash(sequences)

    # 3) Embedding
    embedder = ImmunoBERTAdapter(embedding_dim=cfg.EMBED_DIM)
    embs = embedder.embed_batch(sequences)  # torch tensor [n, dim]

    # 4) Affinity: for demo we compute cosine via GPU routine (also uses embedder)
    #    but keep affinity_kernels for per-pair channels (demonstration)
    affinity_k = AffinityKernels(embedder=embedder)
    # small demo: compute full pairwise cosine similarity on GPU
    sim_matrix = gpu_similarity_matrix(sequences, embedder, batch_size=cfg.BATCH_SIZE)

    # 5) RMT-based sparsification
    adj = rmt_sparsify(sim_matrix, keep_top_fraction=cfg.RMT_KEEP_FRAC)

    # 6) Fair clustering: tune lambda to meet disparity threshold
    lam_opt = tune_lambda_bisection(X=embs.numpy(), subgroup_labels=subgroup_labels,
                                    k=cfg.N_CLUSTERS, delta_max=cfg.DELTA_MAX)

    # Perform fair clustering with found lambda
    labels_pred, centers = fair_kmeans(X=embs.numpy(), subgroup_labels=subgroup_labels,
                                       k=cfg.N_CLUSTERS, lambda_=lam_opt, max_iter=50)

    # 7) Evaluation
    r_at_1 = recall_at_k(adj_matrix=adj, labels=subgroup_labels, k=1)
    purity = purity_score(labels_true=subgroup_labels, labels_pred=labels_pred)
    disparity = clustering_disparity(labels_pred, subgroup_labels)

    print("Demo results:")
    print(f"  lambda_opt: {lam_opt:.4f}")
    print(f"  recall@1 (approx on adjacency): {r_at_1:.4f}")
    print(f"  clustering purity: {purity:.4f}")
    print(f"  clustering disparity (max JS): {disparity:.4f}")

if __name__ == "__main__":
    run_demo()
