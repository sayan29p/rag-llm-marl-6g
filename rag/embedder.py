import numpy as np
from sentence_transformers import SentenceTransformer


class StateEmbedder:
    """Thin wrapper around all-MiniLM-L6-v2 for embedding state descriptions."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Returns
        -------
        np.ndarray, shape (384,), dtype float32
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Returns
        -------
        np.ndarray, shape (N, 384), dtype float32
        """
        return self.model.encode(texts, convert_to_numpy=True)
