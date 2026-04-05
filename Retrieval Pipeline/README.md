# ☁️ Advanced RAG Pipeline: Pinecone Retrieval & Cross-Encoder Evaluation

This project implements a production-ready, two-stage Retrieval-Augmented Generation (RAG) pipeline. It utilizes **Pinecone** for scalable vector storage and introduces a **Cross-Encoder** to mathematically evaluate the true relevance of retrieved documents, preventing irrelevant data from being passed to an LLM.

## 🚀 Architecture Overview

This system utilizes a "Retrieve and Rerank" methodology:
1. **Bi-Encoder (Retrieval):** Uses `all-MiniLM-L6-v2` to quickly fetch the top-K most semantically similar documents from Pinecone.
2. **Cross-Encoder (Evaluation):** Uses `ms-marco-MiniLM-L-6-v2` to read the query and the retrieved documents simultaneously, scoring their actual question-answer relevance.

## 🛠 Tech Stack

* **Backend:** FastAPI (Python)
* **Frontend:** Streamlit
* **Vector Database:** Pinecone (Serverless)
* **Embeddings (Bi-Encoder):** `sentence-transformers/all-MiniLM-L6-v2`
* **Evaluator (Cross-Encoder):** `cross-encoder/ms-marco-MiniLM-L-6-v2`

## ⚙️ Prerequisites

1. Python 3.9+ installed.
2. A free [Pinecone API Key](https://www.pinecone.io/). 

## 📦 Installation & Setup

**1. Install the required dependencies:**
```bash
pip install fastapi uvicorn streamlit sentence-transformers pinecone-client requests python-multipart
```

## Clone or setup your project directory:
Ensure you have api.py and app.py in the same directory.

## Run the application (requires two terminal windows):

### Terminal 1 - Start the Backend API:

Bash
```
python api.py
(Runs on http://localhost:8000)
```

### Terminal 2 - Start the Streamlit UI:

Bash
```
streamlit run app.py
(Runs on http://localhost:8501)
```
