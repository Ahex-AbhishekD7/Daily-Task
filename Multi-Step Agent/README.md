# AI Email Processing Agent

This project is an AI-powered email assistant built with **LangGraph**, **FastAPI** (Backend), and **Streamlit** (Frontend). It uses the Hugging Face `Llama-3.1-8B-Instruct` model to read an email, summarize it, identify the sender's intent, and draft a response.

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python 3.9+ installed. Install the required packages using:
```bash
pip install -r requirements.txt

### 1. Install Dependencies

You need a Hugging Face API token to use the Llama 3.1 model. Set it as an environment variable in your terminal:

Windows (Command Prompt):

```bash
set HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token_here"

## 3. Run the Application
You will need two separate terminal windows to run the backend and frontend simultaneously.

Terminal 1: Start the Backend (FastAPI)

```bash
python api.py
The API will be available at http://localhost:8000

Terminal 2: Start the Frontend (Streamlit)

```bash
streamlit run app.py