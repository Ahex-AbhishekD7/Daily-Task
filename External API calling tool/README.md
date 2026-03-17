**🌤️ AI Weather Tool Agent**


An intelligent "ReAct" (Reason + Act) agent built with LangGraph, FastAPI, and Streamlit. This agent uses a Llama-3.1-8B-Instruct model via Hugging Face to interpret weather queries, handle typos automatically, and fetch real-time data from the OpenWeatherMap API.


**🚀 Features**


Fuzzy Location Correction: Intelligently corrects typos (e.g., "Pairs" → "Paris") before calling the API.

Tool Calling: Integrates directly with OpenWeatherMap using LangChain.

FastAPI Backend: A robust API that manages the agent logic and message filtering.

Streamlit UI: A clean, interactive chat interface for users.

Clean Output: Advanced filtering to ensure internal JSON tool calls are stripped from the final user response.




**🛠️ Tech Stack**


Language: Python 3.13+

Orchestration: LangGraph (v0.4+)

LLM: Meta Llama-3.1-8B-Instruct (via Hugging Face Endpoint)

API Framework: FastAPI

Interface: Streamlit

Tooling: LangChain Community (OpenWeatherMap)


**Install Dependencies:**


Bash
pip install fastapi uvicorn streamlit requests langchain-huggingface langgraph langchain-community pyowm python-dotenv
Environment Setup:
Create a .env file in the root or export your keys:


Bash
export OPENWEATHERMAP_API_KEY='your_openweathermap_api_key'
export HUGGINGFACEHUB_API_TOKEN='your_hf_token'


**🚦 How to Run**


You need to run the Backend and the Frontend in separate terminals.

*1. Start the FastAPI Backend*


Bash
python api.py
The server will start at http://localhost:8000. You can check the docs at /docs.

*2. Start the Streamlit Frontend*


Bash
streamlit run frontend.py
Your browser will open to http://localhost:8501.
