#📄 Local 10-Doc Vector Search (FAISS + FastAPI)
This project is a specialized tool designed to create, store, and search embeddings for exactly 10 documents. It uses a "Brute Force" Flat Index, which is the most accurate method for small datasets, ensuring you get the perfect match for every query.

##🛠 Tech Stack
Backend: FastAPI (Python)

Frontend: Streamlit

Embeddings: sentence-transformers (Model: all-MiniLM-L6-v2)

Vector Store: FAISS (Facebook AI Similarity Search)

Data Persistence: Pickle (for raw text storage)

##🧠 How It Works
Embedding Generation: When you upload 10 files, the all-MiniLM-L6-v2 model converts each text file into a 384-dimensional vector.

Indexing: These vectors are loaded into a FAISS IndexFlatL2. For 10 documents, this provides "Exact Search" (calculating the straight-line distance between your query and every document).

Local Storage: * The mathematical vectors are saved as .faiss.

The actual text content is saved as a .pkl (Pickle) file.

Why? FAISS only stores numbers. We use the Pickle file to "map" the numerical result back to the original text.

##🚀 Setup Instructions

###1. Install Dependencies
Ensure you have Python installed, then run:

Bash
```
pip install fastapi uvicorn streamlit sentence-transformers faiss-cpu requests
```

###2. Project Files


Ensure your directory looks like this:

api.py (The FastAPI server)

app.py (The Streamlit interface)

embeddings/ (Created automatically to store your vectors)

###3. Execution
You need to run two terminals:

Terminal 1 (Backend):

Bash
```
python api.py
```


Terminal 2 (Frontend):

Bash
```
streamlit run app.py
```