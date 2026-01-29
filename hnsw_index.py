# indexing/hnsw_index.py
"""
HNSW / FAISS wrapper for approximate nearest neighbors with graceful fallback.

API:
    index = HNSWIndex(dim, metric='l2', ef_search=50)
    index.add(embeddings)  # numpy array [N, D]
    idxs, dists = index.query(query_embs, k=10)
"""
import numpy as np

class HNSWIndex:
    def __init__(self, dim, metric='cosine', ef_search=50, m=16):
        self.dim = dim
        self.metric = metric
        self.ef_search = ef_search
        self.m = m
        self._impl = None
        # Try faiss
        try:
            import faiss
            self.faiss = faiss
            if metric == 'cosine':
                # faiss expects inner product or L2; for cosine, we normalize vectors and use inner product
                self.index = None
                self._use_faiss = True
            else:
                # L2 index using HNSW
                self.index = faiss.IndexHNSWFlat(dim, m)
                self._use_faiss = True
        except Exception:
            self._use_faiss = False
            self.index = None

        # fallback to sklearn's NearestNeighbors if faiss not available
        if not self._use_faiss:
            from sklearn.neighbors import NearestNeighbors
            self._nn = None

    def add(self, vectors):
        """
        vectors: numpy array [N, D]
        """
        vectors = np.asarray(vectors).astype(np.float32)
        if self._use_faiss:
            if self.metric == 'cosine':
                # normalize vectors
                norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
                vectors = vectors / norms
                # use IndexFlatIP for inner product search
                self.index = self.faiss.IndexFlatIP(self.dim)
                self.index.add(vectors)
            else:
                # direct HNSW usage
                if isinstance(self.index, self.faiss.IndexHNSWFlat):
                    self.index.add(vectors)
                else:
                    self.index = self.faiss.IndexHNSWFlat(self.dim, self.m)
                    self.index.add(vectors)
        else:
            from sklearn.neighbors import NearestNeighbors
            self._nn = NearestNeighbors(n_neighbors=min(50, vectors.shape[0]), algorithm='auto', metric='cosine' if self.metric=='cosine' else 'euclidean')
            self._nn.fit(vectors)
            self._data = vectors

    def query(self, queries, k=10):
        queries = np.asarray(queries).astype(np.float32)
        if self._use_faiss:
            if self.metric == 'cosine':
                # normalize queries
                norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9
                queries = queries / norms
                dists, idxs = self.index.search(queries, k)
                # for inner product, higher is better -> convert to 1 - score to behave like distance if needed
                return idxs, dists
            else:
                dists, idxs = self.index.search(queries, k)
                return idxs, dists
        else:
            dists, idxs = self._nn.kneighbors(queries, n_neighbors=min(k, self._data.shape[0]))
            return idxs, dists
