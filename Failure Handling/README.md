# 🎧 Resilient Customer Support AI Agent

This project implements a highly resilient, production-ready Retrieval-Augmented Generation (RAG) pipeline designed for Customer Support. It features robust failure handling to ensure the agent fails gracefully when external APIs go down or when the knowledge base lacks information.

## ✨ Core Features
* **Conversational UI:** Built with Streamlit's chat elements to simulate a real support widget.
* **Smart Fallbacks (Empty Retrieval):** If no relevant documents are found in the Vector DB, the LLM is bypassed, saving tokens and explicitly preventing hallucinations.
* **Exponential Backoff:** Utilizes the `tenacity` library to automatically retry Hugging Face API calls if rate limits (429) or temporary server errors (500/502) occur.
* **Strict Timeouts:** Enforces a 15-second timeout on LLM inference. If the AI hangs, it catches the error and returns a friendly `504` error to the user rather than freezing the application.

## 🛠 Tech Stack
* **Frontend:** Streamlit
* **Backend:** FastAPI
* **LLM Engine:** Hugging Face Serverless Inference (`HuggingFaceH4/zephyr-7b-beta`)
* **Resilience:** `tenacity`, `httpx`

## 🚀 How to Run

**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**2. Start the Backend API:**
```
Bash
python api.py
```
(The API will start on http://localhost:8000)

**3. Start the Chat Interface:**
Open a new terminal and run:
```
Bash
streamlit run app.py
```
(The UI will open at http://localhost:8501)