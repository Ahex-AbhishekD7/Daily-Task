# api.py
import time
import logging
import uuid
import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

from prompts import SYSTEM_PROMPT, build_user_prompt

# OBSERVABILITY SETUP 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("knowledge_assistant.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# LOCAL VECTOR STORE 
embedder = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
vector_index = faiss.IndexFlatL2(dimension)
document_store = {} # Maps FAISS integer IDs to text chunks

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

def chunk_text(text: str, size: int = 150, overlap: int = 25):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), max(1, size - overlap))]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10), reraise=True)
def call_llm(client: InferenceClient, sys_prompt: str, usr_prompt: str):
    return client.chat_completion(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt}
        ],
        max_tokens=250,
        temperature=0.1
    )

@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    global vector_index, document_store
    
    try:
        # Reset index for fresh session
        vector_index = faiss.IndexFlatL2(dimension)
        document_store = {}
        
        total_chunks = 0
        for file in files:
            content = (await file.read()).decode("utf-8")
            chunks = chunk_text(content)
            
            if not chunks:
                continue
                
            embeddings = embedder.encode(chunks).astype('float32')
            
            start_id = vector_index.ntotal
            vector_index.add(embeddings)
            
            for i, chunk in enumerate(chunks):
                document_store[start_id + i] = {"text": chunk, "filename": file.filename}
                total_chunks += 1
                
        logger.info(f"[INGESTION] Processed {len(files)} files into {total_chunks} chunks.")
        return {"message": f"Successfully indexed {total_chunks} chunks."}
        
    except Exception as e:
        logger.error(f"[INGESTION ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process documents.")

@app.post("/ask")
async def ask_question(request: QueryRequest, hf_key: str = Header(...)):
    start_time = time.time()
    logger.info(f"[REQUEST] '{request.query}'")
    
    if vector_index.ntotal == 0:
        return {"answer": "I cannot answer this based on the provided documents. Please upload documents first.", "sources": []}
        
    try:
        # 1. Retrieval
        query_vector = embedder.encode([request.query]).astype('float32')
        distances, indices = vector_index.search(query_vector, request.top_k)
        
        # 2. Fallback / Thresholding (FAISS L2 distance: lower is more similar. Threshold ~1.5 depending on model)
        L2_THRESHOLD = 1.5 
        valid_chunks = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and dist < L2_THRESHOLD:
                valid_chunks.append(document_store[idx])
                
        if not valid_chunks:
            logger.warning("[RETRIEVAL] Low similarity scores. Triggering fallback.")
            return {"answer": "I cannot answer this based on the provided documents.", "sources": []}
            
        # 3. Prompt Construction
        context = "\n---\n".join([chunk["text"] for chunk in valid_chunks])
        user_prompt = build_user_prompt(context, request.query)
        
        # 4. LLM Generation
        try:
            client = InferenceClient(api_key=hf_key, timeout=15.0)
            llm_start = time.time()
            response = call_llm(client, SYSTEM_PROMPT, user_prompt)
            llm_latency = time.time() - llm_start
            
            answer = response.choices[0].message.content
            
            p_tokens = getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0
            c_tokens = getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0
            
            logger.info(f"[SUCCESS] Latency: {time.time()-start_time:.2f}s (LLM: {llm_latency:.2f}s) | Tokens: {p_tokens} P / {c_tokens} C")
            
            return {
                "answer": answer,
                "sources": list({chunk["filename"] for chunk in valid_chunks}) # Unique filenames
            }
            
        except httpx.TimeoutException:
            logger.error("[TIMEOUT] LLM API timeout.")
            raise HTTPException(status_code=504, detail="AI timed out.")
        except HfHubHTTPError as e:
            logger.error(f"[API_ERROR] {e}")
            raise HTTPException(status_code=502, detail="API limit reached.")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CRITICAL] {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")