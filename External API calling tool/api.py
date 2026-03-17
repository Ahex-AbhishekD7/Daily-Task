import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langgraph.prebuilt import create_react_agent

# --- CONFIG ---
os.environ["OPENWEATHERMAP_API_KEY"] = "WeatherAPIKey"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HuggingFaceAPIKey"

app = FastAPI()

# --- 1. TOOL SETUP ---
weather_wrapper = OpenWeatherMapAPIWrapper()
weather_tool = OpenWeatherMapQueryRun(api_wrapper=weather_wrapper)
tools = [weather_tool]

# --- 2. LLM SETUP ---
llm_engine = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1, 
)
llm = ChatHuggingFace(llm=llm_engine)

# --- 3. UPDATED SYSTEM RULES (Spelling & Cleanup) ---
instructions = (
    "You are a weather assistant with advanced geographical knowledge. "
    "RULES: "
    "1. If a user provides a location with a typo (e.g., 'Pairs', 'Dehli', 'Lonon'), "
    "automatically correct it to the most likely intended city before calling the tool. "
    "2. Use the 'open_weather_map' tool with the CORRECTED city name. "
    "3. Your FINAL response must be ONLY a natural language sentence. "
    "4. NEVER include JSON, tool calls, or code blocks in the final answer. "
    "5. Format: 'The weather in [Corrected City] is currently [Temp] and [Condition].'"
)

# --- 4. AGENT SETUP ---
agent_executor = create_react_agent(llm, tools, prompt=instructions)

class ChatQuery(BaseModel):
    user_input: str

@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    try:
        # result contains the full message history
        result = agent_executor.invoke({"messages": [("user", query.user_input)]})
        
        # --- THE CLEANUP LOGIC ---
        messages = result["messages"]
        final_text = ""

        # Loop backwards to find the actual assistant response
        for m in reversed(messages):
            if m.type == "ai" and m.content.strip():
                # Strip out any lingering JSON artifacts manually
                raw_content = m.content
                if "{" in raw_content:
                    # If the AI accidentally left the tool-call JSON in its final text, 
                    # we split it and take the text that follows the closing bracket.
                    final_text = raw_content.split("}")[-1].strip()
                else:
                    final_text = raw_content.strip()
                
                # If we found a valid sentence, stop
                if final_text:
                    break
        
        return {"answer": final_text or "I couldn't process that location."}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Agent logic failed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)