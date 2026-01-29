# preprocessing/minhash_prefilter.py
"""
Simple MinHash sketch implementation for string sequences.
This is a compact, deterministic MinHash based on multiple randomized hashes.
In real runs you may replace with an optimized/minimized bitset/minhash library.
"""
import numpy as np
import hashlib

class MinHashPrefilter:
    def __init__(self, k=64):
        """
        k: number of hash permutations (sketch size)
        """
        self.k = k
        # We use different salts to emulate independent hash functions
        self.salts = [f"salt{i}".encode() for i in range(k)]

    def _string_hash(self, s, salt):
        # deterministic digest -> integer
        h = hashlib.blake2b(digest_size=8, key=salt)
        h.update(s.encode())
        return int.from_bytes(h.digest(), "big")

    def compute_minhash(self, sequences):
        """
        Compute MinHash sketch for each sequence.
        Returns: numpy array [n_sequences, k] of integers (sketch)
        """
        n = len(sequences)
        sketches = np.zeros((n, self.k), dtype=np.uint64)
        for i, s in enumerate(sequences):
            for j, salt in enumerate(self.salts):
                sketches[i, j] = self._string_hash(s, salt)
        return sketches

    def query_candidates(self, query_seq, sketches, topk=100):
        """
        Naive candidate retrieval: return topk sequences by number of equal hash values.
        sketches: full sketch matrix [n, k]
        """
        q_hashes = np.array([int.from_bytes(__import__("hashlib").blake2b(digest_size=8, key=salt).copy().update(query_seq.encode()) or b'\x00'*8, "big") for salt in self.salts])
        # The above line is intentionally inefficient as a placeholder; use compute_minhash instead
        # For simplicity fallback to compute using same method:
        from hashlib import blake2b
        q_hashes = np.array([int.from_bytes(blake2b(query_seq.encode(), digest_size=8, key=salt).digest(), "big") for salt in self.salts])
        # Count matches
        matches = (sketches == q_hashes).sum(axis=1)
        top_idx = np.argsort(-matches)[:topk]
        return top_idx
