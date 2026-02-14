from sentence_transformers import CrossEncoder

# Cross encoder model (true re-ranking)
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs, top_k=3):
    if not docs:
        return []

    pairs = [(query, d["content"]) for d in docs]

    scores = model.predict(pairs)

    for i, score in enumerate(scores):
        # normalize score to 0–1
        docs[i]["score"] = float(1 / (1 + pow(2.71828, -score)))

    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs[:top_k]
