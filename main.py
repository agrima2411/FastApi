from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from openai import OpenAI

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["OPTIONS", "POST"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI()

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
async def compute_similarity(request: SimilarityRequest):
    # Get embeddings for all documents and query
    doc_embeddings = [get_embedding(doc) for doc in request.docs]
    query_embedding = get_embedding(request.query)
    
    # Calculate similarities
    similarities = [
        cosine_similarity(query_embedding, doc_embedding)
        for doc_embedding in doc_embeddings
    ]
    
    # Get indices of top 3 most similar documents
    top_indices = np.argsort(similarities)[-3:][::-1]
    
    # Return the document contents in order of similarity
    matches = [request.docs[i] for i in top_indices]
    
    return {"matches": matches}