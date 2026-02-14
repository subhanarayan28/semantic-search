from sentence_transformers import SentenceTransformer

# small, fast, good quality model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts):
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
