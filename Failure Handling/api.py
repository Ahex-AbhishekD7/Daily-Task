import logging
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

# Setup logging to track failures
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# --- RETRY LOGIC FOR LLM API ---
# This will retry up to 3 times, waiting 2s, 4s, then 8s between attempts
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True
)
def call_llm_with_retry(client: InferenceClient, system_prompt: str, user_prompt: str):
    logger.info("Attempting to call LLM API...")
    # httpx timeout set to 15 seconds to prevent hanging
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
    try:
       
        # SCENARIO 1: API FAILURE (Vector DB)
        
        try:
            # Mocking Vector DB search for the example
            # search_results = index.query(...) 
            search_results = {'matches': []} # Simulating empty results
        except Exception as db_error:
            logger.error(f"Vector DB connection failed: {db_error}")
            raise HTTPException(status_code=503, detail="Database is temporarily unavailable. Please try again later.")

        # SCENARIO 2: NO DOCUMENT RETRIEVED
        if not search_results.get('matches'):
            logger.info(f"No context found for query: {request.query}")
            return {
                "query": request.query, 
                "answer": "I'm sorry, I couldn't find any information about that in my current knowledge base. Would you like to connect with a human agent?",
                "sources": []
            }
            
        # Construct Context (Assuming matches were found)
        retrieved_texts = [match['metadata']['text'] for match in search_results['matches']]
        context = "\n\n".join(retrieved_texts)
        
        system_prompt = "You are a helpful support agent. Answer strictly based on the context provided."
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {request.query}"

       
        # SCENARIO 3: LLM TIMEOUT & LLM API FAILURES
        
        try:
            client = InferenceClient(api_key=hf_key, timeout=15.0) # Strict 15s timeout
            response = call_llm_with_retry(client, system_prompt, user_prompt)
            answer = response.choices[0].message.content
            
        except httpx.TimeoutException:
            logger.error("LLM API Timeout.")
            raise HTTPException(status_code=504, detail="The AI is taking too long to respond. Please try your question again.")
            
        except HfHubHTTPError as hf_error:
            logger.error(f"LLM API Failure after retries: {hf_error}")
            raise HTTPException(status_code=502, detail="We are experiencing high traffic. Please wait a moment and try again.")

        return {
            "query": request.query, 
            "answer": answer,
            "sources": [{"filename": m['metadata']['filename']} for m in search_results['matches']]
        }
        
    # Catch-all for unexpected crashes
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected Pipeline Error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred processing your request.")