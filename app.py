from fastapi import FastAPI
from pydantic import BaseModel
import time
from vector_store import VectorStore
from reranker import rerank

app = FastAPI()

store = VectorStore()
store.load_docs()
index_ready = False
class QueryRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3

@app.post("/search")
def search(req: QueryRequest):
    global index_ready

    start = time.time()

    # build index only on first request
    if not index_ready:
        store.build_index()
        index_ready = True

    initial = store.search(req.query, req.k)

    reranked = False
    if req.rerank:
        initial = rerank(req.query, initial, req.rerankK)
        reranked = True

    latency = int((time.time() - start) * 1000)

    return {
        "results": initial,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": len(store.docs)
        }
    }
