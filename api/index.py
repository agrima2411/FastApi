import os
import json
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def handler(request):
    try:
        # Parse request body
        body = request.get_json()
        docs = body.get("docs", [])
        query = body.get("query", "")

        if not docs or not query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing docs or query"})
            }

        # Get query embedding
        query_emb = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding

        # Get document embeddings
        doc_embs = [
            client.embeddings.create(input=[d], model="text-embedding-3-small").data[0].embedding
            for d in docs
        ]

        # Compute cosine similarity
        sims = [
            float(np.dot(query_emb, d) / (np.linalg.norm(query_emb) * np.linalg.norm(d)))
            for d in doc_embs
        ]

        best_idx = int(np.argmax(sims))

        return {
            "statusCode": 200,
            "body": json.dumps({
                "best_match": docs[best_idx],
                "similarities": sims
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
