# 🧠 AI Knowledge Assistant

A comprehensive, production-ready RAG system featuring document ingestion, strict fallback mechanisms, structured generation, and full observability.

## 🌟 Core Features
* **Document Upload:** Dynamic ingestion of text documents into a local FAISS vector store for fast, session-based retrieval without cloud database latency.
* **Structured Answers:** Enforces strict point-based formatting (one point per line) for high readability and structured data extraction.
* **Question Answering:** Utilizes `all-MiniLM-L6-v2` for semantic search and `HuggingFaceH4/zephyr-7b-beta` for generation.
* **Fallback Response:** Implements a strict "I don't know" threshold. If document similarity is too low or the context lacks the answer, the LLM safely degrades instead of hallucinating.
* **Observability (Logging):** Comprehensive logging of token usage, latencies, retrieval scores, and pipeline errors to `knowledge_assistant.log`.

## 📂 Project Structure
* `api.py` - FastAPI backend handling embeddings, retrieval, and LLM orchestration.
* `app.py` - Streamlit frontend featuring a document uploader and chat interface.
* `prompts.py` - Isolated prompt templates for easy tuning.
* `requirements.txt` - Dependency list.

## 📝 Usage Flow
1. Open the UI and provide your Hugging Face API key.
2. Upload one or more `.txt` documents via the sidebar.
3. Wait for the success message confirming the documents are chunked and indexed.
4. Ask questions in the chat interface. The assistant will respond strictly with point-based lists backed by your data.

## 🚀 Setup & Execution

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
2.  **Start the API Server:**

    ```bash
    python api.py
    ```
    *(The API runs locally on port 8000)*
3.  **Start the Frontend Interface:**

    ```bash
    streamlit run app.py
    ```
    *(The UI runs locally on port 8501)*