import faiss
import numpy as np
import json
from embedder import embed_text

class VectorStore:
    def __init__(self):
        self.index = None
        self.docs = []

    def load_docs(self):
        with open("docs.json") as f:
            self.docs = json.load(f)

    def build_index(self):
        texts = [d["content"] for d in self.docs]
        embeddings = embed_text(texts)

        dim = len(embeddings[0])
        self.index = faiss.IndexFlatIP(dim)

        vectors = np.array(embeddings).astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(self, query, k=5):
        q_emb = embed_text([query])[0]
        q_emb = np.array([q_emb]).astype("float32")
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.docs[idx]
            results.append({
                "id": doc["id"],
                "score": float(score),
                "content": doc["content"],
                "metadata": {"source": doc["source"]}
            })
        return results
