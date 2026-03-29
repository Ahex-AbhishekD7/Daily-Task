# AI Email Processing Agent

This project is an AI-powered email assistant built with **LangGraph**, **FastAPI** (Backend), and **Streamlit** (Frontend). It uses the Hugging Face `Llama-3.1-8B-Instruct` model to read an email, summarize it, identify the sender's intent, and draft a response.

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python 3.9+ installed. Install the required packages using:
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
You need API keys for both Hugging Face and Pinecone to run the agent. Open the following files and replace the placeholder strings with your actual keys:

Hugging Face: Open api.py and update the HF_TOKEN variable with your token.

Pinecone: Open memory.py and update the PINECONE_API_KEY variable with your Pinecone API key.

### 3. Run the Application
You will need two separate terminal windows to run the backend and frontend simultaneously.

Terminal 1: Start the Backend (FastAPI)

```bash
python api.py
The API will be available at http://localhost:8000
```
Terminal 2: Start the Frontend (Streamlit)

```bash
streamlit run app.py
```
