import os
import time
import logging
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

# OBSERVABILITY SET
# Configure structured logging to write to both the console and a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("agent_observability.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True
)
def call_llm_with_retry(client: InferenceClient, system_prompt: str, user_prompt: str):
    return client.chat_completion(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.1
    )

@app.post("/generate")
async def generate_answer(
    request: QueryRequest, 
    hf_key: str = Header(...)
):
    # 1. Start overall latency timer
    start_time = time.time()
    
    # 2. Log the incoming prompt
    logger.info(f"[REQUEST] New Query Received: '{request.query}'")
    
    try:
        try:
            # Mocking Vector DB search for the example
            search_results = {'matches': []} 
        except Exception as db_error:
            logger.error(f"[ERROR] Vector DB connection failed: {db_error}")
            raise HTTPException(status_code=503, detail="Database is temporarily unavailable.")

        if not search_results.get('matches'):
            logger.warning(f"[RETRIEVAL] No context found for query: '{request.query}'. Bypassing LLM.")
            return {
                "query": request.query, 
                "answer": "I'm sorry, I couldn't find any information about that in my current knowledge base.",
                "sources": []
            }
            
        retrieved_texts = [match['metadata']['text'] for match in search_results['matches']]
        context = "\n\n".join(retrieved_texts)
        
        system_prompt = "You are a helpful support agent. Answer strictly based on the context provided."
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {request.query}"

        try:
            client = InferenceClient(api_key=hf_key, timeout=15.0)
            
            # 3. Start LLM-specific latency timer
            llm_start_time = time.time()
            response = call_llm_with_retry(client, system_prompt, user_prompt)
            llm_latency = time.time() - llm_start_time
            
            answer = response.choices[0].message.content
            
            # 4. Extract token usage (Hugging Face Inference API usually mimics OpenAI's usage object)
            prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else "Unknown"
            completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else "Unknown"
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else "Unknown"
            
            # 5. Log Success Metrics
            total_latency = time.time() - start_time
            logger.info(f"[SUCCESS] Answer Generated. | Total Latency: {total_latency:.2f}s | LLM Latency: {llm_latency:.2f}s | Tokens (P/C/T): {prompt_tokens}/{completion_tokens}/{total_tokens}")
            
        except httpx.TimeoutException:
            logger.error(f"[TIMEOUT] LLM API Timeout after {time.time() - start_time:.2f}s")
            raise HTTPException(status_code=504, detail="The AI is taking too long to respond.")
            
        except HfHubHTTPError as hf_error:
            logger.error(f"[API_ERROR] LLM API Failure: {hf_error}")
            raise HTTPException(status_code=502, detail="We are experiencing high traffic.")

        return {
            "query": request.query, 
            "answer": answer,
            "sources": [{"filename": m['metadata']['filename']} for m in search_results['matches']]
        }
        
    except HTTPException:
        # Re-raise known HTTP exceptions so they don't get caught by the general exception block
        raise
    except Exception as e:
        # 6. Log unexpected errors with latency context
        latency = time.time() - start_time
        logger.error(f"[CRITICAL] Unexpected Pipeline Error after {latency:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred processing your request.")