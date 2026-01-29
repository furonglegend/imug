# embedding/immunobert_adapter.py
"""
Adapter for ImmunoBERT-style embedding. For fast debugging this class
provides a deterministic placeholder embedding based on hashed sequence.
Replace `embed_batch` with a real transformer model call when available.
"""
import torch
import numpy as np

class ImmunoBERTAdapter:
    def __init__(self, embedding_dim=128, device=None):
        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _seq_to_seed_vector(self, seq):
        # Deterministic numeric features from sequence string
        # Map each char to its ASCII code and fold into fixed-size vector
        codes = [ord(c) for c in seq]
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for i, val in enumerate(codes):
            vec[i % self.embedding_dim] += (val & 0xFF) / 255.0
        # Normalize
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec

    def embed_batch(self, seqs):
        """
        Accepts list of strings and returns torch.FloatTensor [n, embedding_dim]
        Placeholder deterministic embeddings; replace with real model inference.
        """
        embs = [self._seq_to_seed_vector(s) for s in seqs]
        embs = torch.from_numpy(np.vstack(embs)).float().to(self.device)
        return embs

    def __call__(self, seq_or_list):
        # Convenience: if list -> embed_batch; if str -> single embedding
        if isinstance(seq_or_list, list):
            return self.embed_batch(seq_or_list)
        elif isinstance(seq_or_list, str):
            return self.embed_batch([seq_or_list])[0]
        else:
            raise ValueError("Input must be str or list of str.")
