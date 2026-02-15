import numpy as np
from embedder import embed_text

def rerank(query, docs, top_k=3):
    if not docs:
        return []

    # embed query again (second scoring stage)
    query_vec = np.array(embed_text([query])[0])

    rescored = []
    for d in docs:
        doc_vec = np.array(embed_text([d["content"]])[0])

        # cosine similarity
        score = np.dot(query_vec, doc_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
        )

        # normalize 0–1
        d["score"] = float((score + 1) / 2)
        rescored.append(d)

    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored[:top_k]
