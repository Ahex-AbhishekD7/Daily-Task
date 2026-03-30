import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

app = FastAPI()

# Load a lightweight model
model = SentenceTransformer('all-MiniLM-L6-v2')
INDEX_PATH = "embeddings/vector_store.faiss"
DOCS_PATH = "embeddings/docs.pkl"

if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

@app.post("/create-embeddings")
async def create_embeddings(files: list[UploadFile] = File(...)):
    try:
        if len(files) != 10:
            raise HTTPException(status_code=400, detail="Please upload exactly 10 documents.")

        documents = []
        for file in files:
            content = await file.read()
            documents.append(content.decode("utf-8"))

        # 1. Generate Embeddings
        embeddings = model.encode(documents)
        
        # 2. Initialize FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))

        # 3. Store Locally
        faiss.write_index(index, INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f)

        return {"status": "success", "message": f"Stored {len(documents)} docs in {INDEX_PATH}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)