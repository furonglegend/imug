# tests/test_affinity.py
"""
Unit tests for affinity kernels and edit-distance utilities.
Run with: pytest tests/test_affinity.py
"""
import numpy as np

from embedding.immunobert_adapter import ImmunoBERTAdapter
from affinity.affinity_kernels import AffinityKernels
from affinity.edit_distance_cpu import levenshtein_two, batch_edit_distance

def test_semantic_cosine_identical_sequences():
    """
    Identical sequences should have cosine similarity close to 1.0
    using the placeholder ImmunoBERTAdapter embeddings.
    """
    embedder = ImmunoBERTAdapter(embedding_dim=64, device="cpu")
    seqs = ["ACGTACGT", "ACGTACGT"]
    embs = embedder.embed_batch(seqs)  # [2, D]
    # compute cosine manually
    a = embs[0].numpy()
    b = embs[1].numpy()
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    assert cos > 0.999, f"expected cos ~1.0, got {cos}"

def test_levenshtein_basic_cases():
    """
    Basic sanity checks of the CPU levenshtein implementation.
    """
    # direct two-string function
    assert levenshtein_two("ABC", "ABC") == 0
    assert levenshtein_two("ABC", "ABD") == 1
    assert levenshtein_two("", "A") == 1
    # batch distances
    a = ["A", "AB", "ABC"]
    b = ["A", "AC"]
    dmat = batch_edit_distance(a, b)
    # expected distances:
    # ["A","AB","ABC"] x ["A","AC"] ->
    # [[0,1],
    #  [1,1],
    #  [2,2]]
    expected = np.array([[0,1],[1,1],[2,2]])
    assert np.array_equal(dmat, expected)

def test_fuse_channels_output_range():
    """
    Fused similarity should lie within [0,1] given semantic and Levenshtein
    similarities in [0,1] and positive weights summing to 1.
    """
    embedder = ImmunoBERTAdapter(embedding_dim=32, device="cpu")
    kernels = AffinityKernels(embedder=embedder, lev_weight=0.4, sem_weight=0.6)
    seqs_a = ["AAAA", "CCCC"]
    seqs_b = ["AAAA", "TTTT"]
    fused = kernels.fuse_channels(seqs_a, seqs_b)  # numpy array [2,2]
    assert fused.shape == (2,2)
    assert np.all(fused >= -1e-6) and np.all(fused <= 1.0001), f"fused out of [0,1]: min {fused.min()}, max {fused.max()}"
