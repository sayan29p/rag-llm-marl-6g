import numpy as np
import faiss
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import EMBEDDING_DIM, RAG_TOP_K


MAX_STORE_SIZE = 10_000   # oldest entries evicted when this limit is reached


class VectorStore:
    """
    FAISS-backed experience store for RAG retrieval.

    Each entry holds:
      - a float32 embedding vector  (shape: EMBEDDING_DIM,)
      - metadata: the serialized state text and the reward achieved
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.index         = faiss.IndexFlatL2(embedding_dim)
        # Parallel list of metadata dicts; index i matches FAISS internal row i.
        # When entries are evicted we rebuild the FAISS index from scratch, so
        # the list and the index always stay aligned.
        self._metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, embedding: np.ndarray, state_text: str, reward: float) -> None:
        """
        Add one experience to the store.

        Parameters
        ----------
        embedding  : np.ndarray, shape (EMBEDDING_DIM,)
        state_text : str    — serialized observation from StateSerializer
        reward     : float  — scalar reward received after that state
        """
        vec = _to_row(embedding)   # (1, D) float32

        # Evict oldest entry if at capacity (before adding the new one)
        if len(self) >= MAX_STORE_SIZE:
            self._evict_oldest()

        self.index.add(vec)
        self._metadata.append({"state_text": state_text, "reward": float(reward)})

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = RAG_TOP_K,
    ) -> list[dict]:
        """
        Find the top_k most similar stored states.

        Parameters
        ----------
        query_embedding : np.ndarray, shape (EMBEDDING_DIM,)
        top_k           : int

        Returns
        -------
        list of dicts, each with keys:
          state_text : str
          reward     : float
          distance   : float   (L2 distance; lower = more similar)
        Sorted by ascending distance (most similar first).
        """
        if len(self) == 0:
            return []

        k = min(top_k, len(self))
        query = _to_row(query_embedding)   # (1, D) float32
        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:          # FAISS returns -1 for unfilled slots
                continue
            meta = self._metadata[idx]
            results.append({
                "state_text": meta["state_text"],
                "reward"    : meta["reward"],
                "distance"  : float(dist),
            })
        return results

    def build_context_string(self, results: list[dict]) -> str:
        """
        Format retrieve() output into a prompt-ready context block.

        Example output
        --------------
        Past similar situation 1: [state text] → reward: -1.23
        Past similar situation 2: [state text] → reward: -0.87
        ...
        """
        if not results:
            return "No similar past situations found."

        lines = []
        for i, entry in enumerate(results, start=1):
            lines.append(
                f"Past similar situation {i}: "
                f"{entry['state_text']} "
                f"→ reward: {entry['reward']:.4f}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.index.ntotal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_oldest(self) -> None:
        """
        Remove the oldest entry (index 0 in _metadata) and rebuild the FAISS
        index from the remaining vectors.

        FAISS IndexFlatL2 does not support in-place removal, so we reconstruct.
        """
        # Reconstruct all stored vectors, drop row 0
        all_vecs = faiss.rev_swig_ptr(
            self.index.get_xb(), len(self) * self.embedding_dim
        ).reshape(len(self), self.embedding_dim).copy()

        remaining_vecs = all_vecs[1:]          # drop oldest
        self._metadata  = self._metadata[1:]   # drop oldest metadata

        self.index.reset()
        if len(remaining_vecs) > 0:
            self.index.add(remaining_vecs.astype(np.float32))


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _to_row(vec: np.ndarray) -> np.ndarray:
    """Ensure vec is a contiguous float32 row matrix of shape (1, D)."""
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    return np.ascontiguousarray(vec)
