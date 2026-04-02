# 📄 Local 10-Doc Vector Search (FAISS + FastAPI)

A lightweight semantic search system designed specifically for **exactly 10 documents**.
This project uses a **brute-force FAISS index** to guarantee **maximum accuracy**, making it ideal for small datasets where precision matters more than speed.


## 🚀 Features

* 🔍 Exact similarity search using FAISS `IndexFlatL2`
* 🧠 High-quality embeddings via `all-MiniLM-L6-v2`
* ⚡ Fast API backend with FastAPI
* 🎨 Simple UI using Streamlit
* 💾 Persistent storage using FAISS + Pickle
* 📂 Designed specifically for **10 documents only**


## 🛠 Tech Stack

| Component    | Technology                                 |
| ------------ | ------------------------------------------ |
| Backend      | FastAPI (Python)                           |
| Frontend     | Streamlit                                  |
| Embeddings   | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS                                      |
| Storage      | Pickle                                     |


## 🧠 How It Works

### 1. Embedding Generation

* Upload 10 documents
* Each document is converted into a **384-dimensional vector** using:

  ```
  all-MiniLM-L6-v2
  ```

### 2. Indexing

* Vectors are stored in:

  ```
  FAISS IndexFlatL2
  ```
* This performs **exact (brute-force) similarity search**
* It calculates distance between your query and **every document**

### 3. Local Storage

* `.faiss` → stores vector data
* `.pkl` → stores original text

👉 Why both?

* FAISS only stores numbers (vectors)
* Pickle maps results back to readable text


## 📁 Project Structure

```
project-root/
│
├── api.py          # FastAPI backend
├── app.py          # Streamlit frontend
├── embeddings/     # Auto-created vector storage
└── README.md
```


## ⚙️ Setup Instructions

### 1. Install Dependencies

Make sure Python is installed, then run:

```bash
pip install fastapi uvicorn streamlit sentence-transformers faiss-cpu requests
```


### 2. Run the Application

You need **two terminals**:

#### ▶️ Terminal 1 — Start Backend

```bash
python api.py
```

#### ▶️ Terminal 2 — Start Frontend

```bash
streamlit run app.py
```
