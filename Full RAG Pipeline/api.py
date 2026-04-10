import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# Setup basic logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
INDEX_NAME = "optimized-rag-pipeline"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5  # Increased default for better context
    chunk_size: int = 150 # Words per chunk
    chunk_overlap: int = 25 # Overlap to maintain context between chunks

def get_overlapping_chunks(text: str, chunk_size: int, overlap: int):
    """Splits text into overlapping word chunks to preserve semantic meaning."""
    words = text.split()
    chunks = []
    if not words:
        return chunks
        
    for i in range(0, len(words), max(1, chunk_size - overlap)):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

@app.post("/upsert")
async def upsert_documents(
    files: list[UploadFile] = File(...), 
    pinecone_key: str = Header(...)
):
    try:
        pc = Pinecone(api_key=pinecone_key)
        
        # Initialize Index safely
        if INDEX_NAME not in pc.list_indexes().names():
            logger.info(f"Creating new index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        index = pc.Index(INDEX_NAME)
        
        vectors_to_upsert = []
        total_chunks = 0
        
        for file in files:
            content = (await file.read()).decode("utf-8")
            if not content.strip():
                continue # Skip empty files
                
            # Create overlapping chunks
            chunks = get_overlapping_chunks(content, chunk_size=150, overlap=25)
            total_chunks += len(chunks)
            
            for i, chunk_text in enumerate(chunks):
                vector = embedder.encode(chunk_text).tolist()
                doc_id = f"{file.filename}-chunk-{i}-{str(uuid.uuid4())[:8]}"
                
                vectors_to_upsert.append({
                    "id": doc_id, 
                    "values": vector, 
                    "metadata": {"text": chunk_text, "filename": file.filename, "chunk_index": i}
                })
        
        if not vectors_to_upsert:
            raise HTTPException(status_code=400, detail="No readable text found in uploaded files.")
            
        # Batch upsert to prevent timeout on large files
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            index.upsert(vectors=vectors_to_upsert[i:i + batch_size])
            
        return {"message": f"Success! Processed {len(files)} files into {total_chunks} searchable chunks."}
        
    except Exception as e:
        logger.error(f"Upsert failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")

@app.post("/generate")
async def generate_answer(
    request: QueryRequest, 
    pinecone_key: str = Header(...),
    hf_key: str = Header(...)
):
    try:
        # 1. Retrieval Stage
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(INDEX_NAME)
        
        query_vector = embedder.encode(request.query).tolist()
        search_results = index.query(
            vector=query_vector, 
            top_k=request.top_k, 
            include_metadata=True
        )
        
        if not search_results['matches']:
            return {"query": request.query, "answer": "No relevant documents found.", "sources": []}
            
        # 2. Context Construction Stage
        retrieved_texts = []
        sources = []
        
        for match in search_results['matches']:
            meta = match['metadata']
            retrieved_texts.append(f"[Source: {meta['filename']}]\n{meta['text']}")
            sources.append({
                "filename": meta['filename'], 
                "chunk": meta['chunk_index'],
                "score": match['score']
            })
            
        context = "\n\n---\n\n".join(retrieved_texts)
        
        # 3. Optimized Prompting Stage
        system_prompt = """You are an expert, highly accurate data assistant. Your ONLY job is to answer questions based on the provided context.
        
        RULES:
        1. If the context contains the answer, explain it clearly and cite the source filename.
        2. If the context partially answers the question, provide what you know and explicitly state what is missing.
        3. If the context DOES NOT contain the answer, reply EXACTLY with: "I cannot answer this based on the provided documents."
        4. Do NOT hallucinate or use outside knowledge."""
        
        user_prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION: {request.query}\n\nANSWER STRICTLY BASED ON CONTEXT:"

        # 4. Generation Stage (with HF rate limit handling)
        try:
            client = InferenceClient(api_key=hf_key)
            response = client.chat_completion(
                model="HuggingFaceH4/zephyr-7b-beta",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.1 # Very low temperature for highly deterministic, factual answers
            )
            answer = response.choices[0].message.content
            
        except HfHubHTTPError as e:
            logger.error(f"HuggingFace API Error: {str(e)}")
            raise HTTPException(status_code=502, detail="HuggingFace API rate limit or model timeout. Please try again in a few seconds.")
        
        return {
            "query": request.query, 
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
