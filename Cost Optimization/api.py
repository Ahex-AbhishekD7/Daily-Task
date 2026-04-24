import time
import logging
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

# --- OBSERVABILITY SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("agent_observability.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 2  # Reduced from 3 to save tokens

# --- COST OPTIMIZATION: SYSTEM PROMPT COMPRESSION ---
# Previous: 50+ tokens. New: ~25 tokens.
COMPRESSED_SYSTEM_PROMPT = "You are a support bot. Answer using ONLY the context below. If missing, say 'I don't know'."

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10), reraise=True)
def call_llm_with_retry(client: InferenceClient, system_prompt: str, user_prompt: str):
    return client.chat_completion(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=150, # Cost Optimization: Cap output tokens strictly
        temperature=0.1
    )

@app.post("/generate")
async def generate_answer(request: QueryRequest, hf_key: str = Header(...)):
    start_time = time.time()
    logger.info(f"[REQUEST] Query: '{request.query}'")
    
    try:
        try:
            # Mocking Vector DB search with dummy scores for demonstration
            # In production: search_results = index.query(...) 
            search_results = {'matches': [
                {'metadata': {'text': 'Password resets can be done in the settings menu.', 'filename': 'doc1.txt'}, 'score': 0.85},
                {'metadata': {'text': 'We offer a 30 day refund policy.', 'filename': 'doc2.txt'}, 'score': 0.30}
            ]} 
        except Exception as db_error:
            logger.error(f"[ERROR] DB failed: {db_error}")
            raise HTTPException(status_code=503, detail="Database unavailable.")

        # --- COST OPTIMIZATION: CONTEXT FILTERING ---
        # Only keep chunks with a high similarity score.
        SIMILARITY_THRESHOLD = 0.5
        filtered_matches = [m for m in search_results.get('matches', []) if m.get('score', 0) > SIMILARITY_THRESHOLD]

        if not filtered_matches:
            logger.warning(f"[RETRIEVAL] No context > {SIMILARITY_THRESHOLD}. Bypassing LLM.")
            return {"query": request.query, "answer": "I don't know based on the knowledge base.", "sources": []}
            
        retrieved_texts = [match['metadata']['text'] for match in filtered_matches]
        context = "\n".join(retrieved_texts)
        
        user_prompt = f"Ctx:{context}\nQ:{request.query}"

        try:
            client = InferenceClient(api_key=hf_key, timeout=10.0)
            llm_start_time = time.time()
            response = call_llm_with_retry(client, COMPRESSED_SYSTEM_PROMPT, user_prompt)
            llm_latency = time.time() - llm_start_time
            
            answer = response.choices[0].message.content
            
            prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0
            completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
            
            logger.info(f"[SUCCESS] Total Latency: {time.time() - start_time:.2f}s | LLM: {llm_latency:.2f}s | Tokens(P/C/T): {prompt_tokens}/{completion_tokens}/{total_tokens}")
            
            return {
                "query": request.query, 
                "answer": answer,
                "sources": [{"filename": m['metadata']['filename'], "score": m['score']} for m in filtered_matches],
                "telemetry": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
            }
            
        except httpx.TimeoutException:
            logger.error("[TIMEOUT] LLM API Timeout")
            raise HTTPException(status_code=504, detail="AI timed out.")
        except HfHubHTTPError as hf_error:
            logger.error(f"[API_ERROR] Failure: {hf_error}")
            raise HTTPException(status_code=502, detail="API limit reached.")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CRITICAL] Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal error.")