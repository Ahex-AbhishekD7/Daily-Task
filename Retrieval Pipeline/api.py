import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone, ServerlessSpec
import uuid

app = FastAPI()

# 1. Load Models
# Bi-encoder for creating vectors (Dimension: 384)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# Cross-encoder for evaluating relevance
evaluator = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

INDEX_NAME = "retrieval-pipeline"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/upsert")
async def upsert_documents(
    files: list[UploadFile] = File(...), 
    pinecone_key: str = Header(...)
):
    try:
        pc = Pinecone(api_key=pinecone_key)
        
        # Create index if it doesn't exist
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        index = pc.Index(INDEX_NAME)
        
        # Read and embed documents
        vectors_to_upsert = []
        for file in files:
            content = (await file.read()).decode("utf-8")
            vector = embedder.encode(content).tolist()
            doc_id = str(uuid.uuid4())
            
            # Pinecone stores the vector and the raw text in metadata
            vectors_to_upsert.append({
                "id": doc_id, 
                "values": vector, 
                "metadata": {"text": content, "filename": file.filename}
            })
            
        index.upsert(vectors=vectors_to_upsert)
        return {"message": f"Successfully upserted {len(vectors_to_upsert)} documents to Pinecone."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve_and_evaluate(
    request: QueryRequest, 
    pinecone_key: str = Header(...)
):
    try:
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(INDEX_NAME)
        
        # 1. Search Vector DB
        query_vector = embedder.encode(request.query).tolist()
        search_results = index.query(
            vector=query_vector, 
            top_k=request.top_k, 
            include_metadata=True
        )
        
        # 2. Evaluate Retrieval Relevance (Cross-Encoder)
        evaluated_results = []
        for match in search_results['matches']:
            doc_text = match['metadata']['text']
            
            # The Cross-Encoder scores how well the doc answers the query
            relevance_score = evaluator.predict([request.query, doc_text])
            
            evaluated_results.append({
                "id": match['id'],
                "filename": match['metadata']['filename'],
                "text": doc_text,
                "pinecone_similarity": match['score'],
                "relevance_score": float(relevance_score)
            })
            
        # Sort by actual relevance rather than just vector similarity
        evaluated_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {"query": request.query, "results": evaluated_results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)