# gpu/similarity_matrix.py
"""
Batched GPU (or CPU fallback) similarity matrix computation using embeddings.
This implementation computes pairwise cosine similarities in blocks to limit memory.
"""
import torch
import numpy as np
import torch.nn.functional as F

def gpu_similarity_matrix(sequences, immunoBERT_model, batch_size=256):
    """
    sequences: list[str]
    immunoBERT_model: object with embed_batch(list[str]) -> torch.Tensor
    batch_size: block size for tiling
    Returns: full similarity matrix as numpy array [n, n]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(sequences)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(0, n, batch_size):
        batch_i = sequences[i:i+batch_size]
        embs_i = immunoBERT_model.embed_batch(batch_i).to(device)
        embs_i = F.normalize(embs_i.float(), p=2, dim=1)
        for j in range(0, n, batch_size):
            batch_j = sequences[j:j+batch_size]
            embs_j = immunoBERT_model.embed_batch(batch_j).to(device)
            embs_j = F.normalize(embs_j.float(), p=2, dim=1)
            # pairwise dot product
            with torch.no_grad():
                block = torch.matmul(embs_i, embs_j.t()).cpu().numpy()
            sim[i:i+len(batch_i), j:j+len(batch_j)] = block
    return sim
