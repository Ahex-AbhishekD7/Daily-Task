from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import StateGraph, START, END

# Import updated prompts
from prompt import summarize_prompt, intent_prompt, urgency_prompt, smart_reply_prompt, draft_prompt
from memory import PineconeMemory

app = FastAPI(title="Email Processing Agent API")

HF_TOKEN = "HUGGINGFACE_API_TOKEN"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

memory_store = PineconeMemory()

llm_engine = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1, 
    huggingfacehub_api_token=HF_TOKEN
)
llm = ChatHuggingFace(llm=llm_engine)

# --- 1. UPDATED STATE ---
class EmailState(TypedDict):
    user_id: str
    email_content: str
    past_context: str
    summary: str
    intent: str
    urgency: str           # NEW
    smart_replies: str     # NEW
    draft: str

# --- 2. NODES ---
def retrieve_context(state: EmailState):
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

# NEW NODE: Detect Urgency
def detect_urgency(state: EmailState):
    chain = urgency_prompt | llm
    result = chain.invoke({"email": state["email_content"]})
    # Clean up output to ensure it's just the word
    clean_urgency = result.content.strip().replace("'", "").replace('"', '')
    return {"urgency": clean_urgency}

# NEW NODE: Generate Smart Replies
def generate_smart_replies(state: EmailState):
    chain = smart_reply_prompt | llm
    result = chain.invoke({"email": state["email_content"]})
    return {"smart_replies": result.content}

def draft_response(state: EmailState):
    chain = draft_prompt | llm
    result = chain.invoke({
        "email": state["email_content"], 
        "intent": state["intent"],
        "urgency": state["urgency"], # Passing urgency to the drafter
        "past_context": state["past_context"]
    })
    return {"draft": result.content}

def save_to_memory(state: EmailState):
    memory_text = f"Email: {state['email_content']}\nSummary: {state['summary']}"
    memory_store.add_memory(state["user_id"], memory_text)
    return {}

# --- 3. BUILD UPDATED GRAPH ---
workflow = StateGraph(EmailState)

workflow.add_node("context_retriever", retrieve_context)
workflow.add_node("summarizer", summarize_email)
workflow.add_node("intent_identifier", identify_intent)
workflow.add_node("urgency_detector", detect_urgency)            # Added
workflow.add_node("smart_reply_generator", generate_smart_replies) # Added
workflow.add_node("drafter", draft_response)
workflow.add_node("memory_saver", save_to_memory)

workflow.add_edge(START, "context_retriever")
workflow.add_edge("context_retriever", "summarizer")
workflow.add_edge("summarizer", "intent_identifier")
workflow.add_edge("intent_identifier", "urgency_detector")
workflow.add_edge("urgency_detector", "smart_reply_generator")
workflow.add_edge("smart_reply_generator", "drafter")
workflow.add_edge("drafter", "memory_saver")
workflow.add_edge("memory_saver", END)

app_graph = workflow.compile()

# --- 4. API ENDPOINTS ---
class EmailRequest(BaseModel):
    user_id: str = "default_user"
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
            "urgency": "",
            "smart_replies": "",
            "draft": ""
        }
        
        result = app_graph.invoke(initial_state)
        
        return {
            "summary": result.get("summary", ""),
            "intent": result.get("intent", ""),
            "urgency": result.get("urgency", "Medium"),
            "smart_replies": result.get("smart_replies", ""),
            "draft": result.get("draft", ""),
            "past_context_used": result.get("past_context", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
