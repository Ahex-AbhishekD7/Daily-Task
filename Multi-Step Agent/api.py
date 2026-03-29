from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import StateGraph, START, END

# Import separated logic
from prompt import summarize_prompt, intent_prompt, draft_prompt
from memory import PineconeMemory

# APP & API KEY SETUP
app = FastAPI(title="Email Processing Agent API")

hf_token = os.getenv("HF_TOKEN") or "HF_TOKEN"

# Initialize Pinecone Memory
memory_store = PineconeMemory()

# LLM SETUP 
llm_engine = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3,
    huggingfacehub_api_token=hf_token 
)
llm = ChatHuggingFace(llm=llm_engine)

# LANGGRAPH STATE 
class EmailState(TypedDict):
    user_id: str
    email_content: str
    past_context: str
    summary: str
    intent: str
    draft: str

# LANGGRAPH NODES 
def retrieve_context(state: EmailState):
    """Fetches related past emails from Pinecone."""
    memories = memory_store.search_memory(state["user_id"], state["email_content"])
    context_str = "\n---\n".join(memories) if memories else "No relevant past emails found."
    return {"past_context": context_str}

def summarize_email(state: EmailState):
    chain = summarize_prompt | llm
    result = chain.invoke({"email": state["email_content"]})
    return {"summary": result.content}

def identify_intent(state: EmailState):
    chain = intent_prompt | llm
    result = chain.invoke({"email": state["email_content"]})
    return {"intent": result.content}

def draft_response(state: EmailState):
    chain = draft_prompt | llm
    result = chain.invoke({
        "email": state["email_content"], 
        "intent": state["intent"],
        "past_context": state["past_context"]
    })
    return {"draft": result.content}

def save_to_memory(state: EmailState):
    """Saves the processed email into Pinecone for future reference."""
    # We save the email content along with its summary for rich context
    memory_text = f"Email: {state['email_content']}\nSummary: {state['summary']}"
    memory_store.add_memory(state["user_id"], memory_text)
    return {} # No state updates needed here

# BUILD GRAPH 
workflow = StateGraph(EmailState)

workflow.add_node("context_retriever", retrieve_context)
workflow.add_node("summarizer", summarize_email)
workflow.add_node("intent_identifier", identify_intent)
workflow.add_node("drafter", draft_response)
workflow.add_node("memory_saver", save_to_memory)

# Define the flow
workflow.add_edge(START, "context_retriever")
workflow.add_edge("context_retriever", "summarizer")
workflow.add_edge("summarizer", "intent_identifier")
workflow.add_edge("intent_identifier", "drafter")
workflow.add_edge("drafter", "memory_saver")
workflow.add_edge("memory_saver", END)

app_graph = workflow.compile()

# API ENDPOINTS 
class EmailRequest(BaseModel):
    user_id: str = "default_user" # Hardcoded default for testing
    email_text: str

@app.post("/process-email")
async def process_email(request: EmailRequest):
    if not request.email_text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")
    
    try:
        initial_state = {
            "user_id": request.user_id,
            "email_content": request.email_text,
            "past_context": "",
            "summary": "",
            "intent": "",
            "draft": ""
        }
        
        result = app_graph.invoke(initial_state)
        
        return {
            "summary": result.get("summary", ""),
            "intent": result.get("intent", ""),
            "draft": result.get("draft", ""),
            "past_context_used": result.get("past_context", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
