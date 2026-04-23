# 📊 Observability Implementation for Customer Support AI Agent

* This module adds comprehensive observability to the Customer Support RAG pipeline.
* It tracks system performance, monitors LLM API usage, and logs pipeline errors.
* The system utilizes Python's built-in logging module to write structured logs to both the console and a local file.

## 🔍 Logged Metrics

* **Prompts:** Captures the exact user query immediately upon receiving the request (`[REQUEST]`).
* **Tokens:** Extracts and logs Prompt, Completion, and Total token usage directly from the Hugging Face API response (`[SUCCESS]`).
* **Latency:** Measures overall API latency and isolated LLM inference latency in seconds.
* **Errors:** Captures vector DB failures, network timeouts (`[TIMEOUT]`), rate limits (`[API_ERROR]`), and unexpected Python crashes (`[CRITICAL]`).
* **Retrieval Bypasses:** Logs warnings (`[WARNING]`) when the LLM is bypassed due to empty context retrieval.

## 📂 Output

* All logs are automatically written to a local file named `agent_observability.log` in the root directory.
* Logs are formatted with timestamps and severity levels for easy parsing and debugging.

## Start the Backend API:

```Bash
python api.py
```
(The API will start on http://localhost:8000)

## Start the Chat Interface:
Open a new terminal and run:

```Bash
streamlit run app.py
```
(The UI will open at http://localhost:8501)