# 🤖 Full RAG Pipeline: Pinecone + LLM

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system. It combines local embedding generation, cloud vector storage (Pinecone), and large language model (LLM) text synthesis to answer questions based strictly on uploaded documents.

## 🚀 The Pipeline Flow
1. **User Query:** The user inputs a question in the Streamlit UI.
2. **Retrieve Docs:** The FastAPI backend embeds the query using `sentence-transformers` and searches Pinecone for the top-k most relevant document chunks via cosine similarity.
3. **Send to LLM:** The backend constructs a strict prompt containing the retrieved text chunks and the user's query, instructing the LLM to answer *only* using the provided context.
4. **Generate Answer:** The OpenAI API processes the context and generates a synthesized, human-readable response, which is sent back to the UI alongside the source citations.

## 🛠 Tech Stack
* **Frontend:** Streamlit
* **Backend:** FastAPI
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Local Bi-Encoder)
* **Vector Database:** Pinecone
* **LLM:** OpenAI (`gpt-3.5-turbo`)

## ⚙️ Prerequisites
* Python 3.9+
* A [Pinecone API Key](https://www.pinecone.io/)
* An [OpenAI API Key](https://platform.openai.com/)

## 📦 Installation & Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the API (Terminal 1):**

```Bash
python api.py
```
3. **Start the UI (Terminal 2):**

```Bash
streamlit run app.py
```

# 📖 Usage

1. Open the UI at http://localhost:8501.

2. Input your Pinecone and OpenAI API keys in the sidebar.

3. Upload your .txt files in the "Populate Vector Database" section to index them.

4. Ask a question in the text box. The LLM will generate an answer and cite the source files it used.
