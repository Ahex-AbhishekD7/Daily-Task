from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

# --- 1. APP & MODEL SETUP ---
app = FastAPI(title="Email Processing Agent API")

# Ensure the Hugging Face token is set in your environment
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    print("Warning: HUGGINGFACEHUB_API_TOKEN is not set in the environment.")

# --- 2. LLM SETUP (From your prompt) ---
llm_engine = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1, 
)
llm = ChatHuggingFace(llm=llm_engine)

# --- 3. LANGGRAPH STATE ---
class EmailState(TypedDict):
    email_content: str
    summary: str
    intent: str
    draft: str

# --- 4. LANGGRAPH NODES ---
def summarize_email(state: EmailState):
    prompt = PromptTemplate.from_template("Summarize the following email concisely:\n\n{email}")
    chain = prompt | llm
    result = chain.invoke({"email": state["email_content"]})
    return {"summary": result.content}

def identify_intent(state: EmailState):
    prompt = PromptTemplate.from_template("Identify the primary intent of this email (e.g., inquiry, complaint, meeting request, spam). Keep it to 1-3 words.\n\nEmail:\n{email}")
    chain = prompt | llm
    result = chain.invoke({"email": state["email_content"]})
    return {"intent": result.content}

def draft_response(state: EmailState):
    prompt = PromptTemplate.from_template(
        "Draft a professional and polite response to the following email. "
        "The identified intent is '{intent}'. "
        "Make sure the response directly addresses the intent.\n\n"
        "Original Email:\n{email}\n\nDraft Response:"
    )
    chain = prompt | llm
    result = chain.invoke({"email": state["email_content"], "intent": state["intent"]})
    return {"draft": result.content}

# --- 5. BUILD GRAPH ---
workflow = StateGraph(EmailState)

workflow.add_node("summarizer", summarize_email)
workflow.add_node("intent_identifier", identify_intent)
workflow.add_node("drafter", draft_response)

workflow.add_edge(START, "summarizer")
workflow.add_edge("summarizer", "intent_identifier")
workflow.add_edge("intent_identifier", "drafter")
workflow.add_edge("drafter", END)

app_graph = workflow.compile()

# --- 6. API ENDPOINTS ---
class EmailRequest(BaseModel):
    email_text: str

@app.post("/process-email")
async def process_email(request: EmailRequest):
    if not request.email_text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")
    
    try:
        # Run the LangGraph agent
        initial_state = {"email_content": request.email_text}
        result = app_graph.invoke(initial_state)
        
        return {
            "summary": result.get("summary", ""),
            "intent": result.get("intent", ""),
            "draft": result.get("draft", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)