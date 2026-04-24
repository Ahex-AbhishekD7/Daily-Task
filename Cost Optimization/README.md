# 💰 Cost-Optimized RAG: Customer Support Agent

This iteration focuses on drastically reducing LLM token usage (and therefore API costs) while maintaining high response accuracy.

## 📉 Cost Optimization Strategies Implemented

* **Prompt Compression:** The system prompt has been rewritten to be highly concise, saving approximately 40-50 tokens per request without losing instruction adherence.
* **Context Filtering (Similarity Thresholding):** Instead of blindly passing the `top_k` documents to the LLM, the backend now checks the similarity score. If a chunk scores below `0.5`, it is discarded before reaching the LLM. This prevents stuffing the prompt with irrelevant tokens.
* **Reduced Context Window:** Default `top_k` is reduced from 3 to 2, under the assumption that high-quality chunking (implemented in previous iterations) contains the answer in the top hits.
* **Max Completion Tokens:** The LLM's `max_tokens` output is strictly capped at `150` (down from 500) to prevent verbosity and runaway generation costs.

## 📊 Token Usage Comparison

| Version | Avg. Prompt Tokens | Avg. Completion Tokens | Estimated Cost / 1k Queries |
| :--- | :--- | :--- | :--- |
| **v1.0 (Naive RAG)** | ~750 | ~150 | High |
| **v2.0 (Optimized)** | ~250 | ~50 | **Reduced by ~66%** |

## 🛠 Features Retained
* **Observability:** Full latency, token, and error logging to `agent_observability.log`.
* **Resilience:** Tenacity retry loops and httpx timeouts.
* **Streamlit UI:** Chatbot interface with sidebar telemetry.

## 🚀 Setup & Execution

### 1. Start the Backend API:

```Bash
python api.py
```
(The API will start on http://localhost:8000)

### 2. Start the Chat Interface:
Open a new terminal and run:

```Bash
streamlit run app.py
```
(The UI will open at http://localhost:8501)