# affinity/affinity_kernels.py
"""
Compute multiple affinity channels between sequences:
 - semantic (cosine on embeddings)
 - normalized Levenshtein similarity
 - optional metadata similarity (simple equality or numeric distance)
Expose a convenience function that fuses channels via simple weighted sum.
"""
import numpy as np
import torch
import torch.nn.functional as F
from .edit_distance_cpu import batch_edit_distance

class AffinityKernels:
    def __init__(self, embedder, lev_weight=0.5, sem_weight=0.5):
        """
        embedder: object with embed_batch(list[str]) -> torch.Tensor
        weights: default fusion weights for channels
        """
        self.embedder = embedder
        self.lev_weight = lev_weight
        self.sem_weight = sem_weight

    def semantic_cosine(self, seqs_a, seqs_b):
        """
        Return cosine similarity matrix between two lists of sequences using embeddings.
        output shape: [len(seqs_a), len(seqs_b)]
        """
        embs_a = self.embedder.embed_batch(seqs_a)
        embs_b = self.embedder.embed_batch(seqs_b)
        # compute pairwise cosine via matrix product
        embs_a_n = F.normalize(embs_a, p=2, dim=1)
        embs_b_n = F.normalize(embs_b, p=2, dim=1)
        sim = torch.matmul(embs_a_n, embs_b_n.t()).cpu().numpy()
        return sim

    def levenshtein_similarity(self, seqs_a, seqs_b):
        """
        Compute normalized Levenshtein similarity in [0,1]: 1 - (edit_distance / max_len)
        Returns numpy array [len(a), len(b)]
        """
        dists = batch_edit_distance(seqs_a, seqs_b)
        maxlen = np.maximum(np.maximum.outer([len(s) for s in seqs_a], [len(s) for s in seqs_b]), 1)
        sim = 1.0 - (dists / maxlen)
        return sim

    def fuse_channels(self, seqs_a, seqs_b, alpha_sem=None, alpha_lev=None):
        """
        Compute fused similarity as weighted sum of semantic and Levenshtein channels.
        If alphas omitted use instance defaults.
        """
        if alpha_sem is None: alpha_sem = self.sem_weight
        if alpha_lev is None: alpha_lev = self.lev_weight
        sem = self.semantic_cosine(seqs_a, seqs_b)
        lev = self.levenshtein_similarity(seqs_a, seqs_b)
        fused = alpha_sem * sem + alpha_lev * lev
        return fused
