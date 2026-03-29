import numpy as np
from rag.state_serializer import StateSerializer
from rag.embedder import StateEmbedder
from rag.vector_store import VectorStore

obs = np.random.rand(170).astype(np.float32)
ser = StateSerializer()
text = ser.serialize(obs)
print("Serialized text length:", len(text), "chars")
print(text[:300])
print()

emb = StateEmbedder()
vec = emb.embed(text)
print("Embedding shape:", vec.shape)
print()

store = VectorStore()
store.add(vec, text, reward=-1.5)
store.add(vec, text, reward=-0.8)
results = store.retrieve(vec, top_k=2)
print("Retrieved", len(results), "results")
print(store.build_context_string(results)[:200])
print()
print("RAG PIPELINE WORKS!")
