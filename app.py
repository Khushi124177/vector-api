import os
import math
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS enable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

async def get_embedding(text: str):
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": "text-embedding-3-small", "input": text}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

@app.post("/similarity")
async def similarity(req: SimilarityRequest):
    query_emb = await get_embedding(req.query)

    scores = []
    for doc in req.docs:
        doc_emb = await get_embedding(doc)
        sim = cosine_similarity(query_emb, doc_emb)
        scores.append((sim, doc))

    scores.sort(reverse=True, key=lambda x: x[0])
    top3 = [doc for _, doc in scores[:3]]

    return {"matches": top3}