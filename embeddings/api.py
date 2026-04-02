import os
import pickle
import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

app = FastAPI()

# Model and Storage Config
model = SentenceTransformer('all-MiniLM-L6-v2')
STORE_DIR = "embeddings"
INDEX_PATH = os.path.join(STORE_DIR, "vector_store.faiss")
DOCS_PATH = os.path.join(STORE_DIR, "docs.pkl")

if not os.path.exists(STORE_DIR):
    os.makedirs(STORE_DIR)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 1

@app.post("/create-embeddings")
async def create_embeddings(files: list[UploadFile] = File(...)):
    try:
        if len(files) != 10:
            raise HTTPException(status_code=400, detail="Please upload exactly 10 documents.")

        documents = []
        for file in files:
            content = await file.read()
            documents.append(content.decode("utf-8"))

        # Generate Embeddings (Shape: 10, 384)
        embeddings = model.encode(documents).astype('float32')
        
        # FAISS Setup for small datasets
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Exact L2 search
        index.add(embeddings)

        # Save locally
        faiss.write_index(index, INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f)

        return {"status": "success", "message": "10 documents indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/search")
async def search_docs(data: SearchQuery):
    try:
        if not os.path.exists(INDEX_PATH):
            raise HTTPException(status_code=404, detail="No index found. Please upload docs first.")

        # Load Index and Docs
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            stored_docs = pickle.load(f)

        # Vectorize Search Query
        query_vector = model.encode([data.query]).astype('float32')
        
        # FAISS Search
        distances, indices = index.search(query_vector, data.top_k)
        
        # Map indices back to text
        results = [{"text": stored_docs[idx], "score": float(distances[0][i])} 
                   for i, idx in enumerate(indices[0])]

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
