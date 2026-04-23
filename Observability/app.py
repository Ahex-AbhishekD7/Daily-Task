import streamlit as st
import requests
import os
import time

st.set_page_config(page_title="Observable AI Agent", layout="wide", page_icon="📊")
st.title("🎧 Customer Support AI")
st.caption("Production-Ready RAG with Built-In Observability")

# CONFIGURATION & OBSERVABILITY 
st.sidebar.header("🔑 Configuration")
hf_key = st.sidebar.text_input("Hugging Face Token", type="password")

st.sidebar.divider()

st.sidebar.header("📊 System Observability")
st.sidebar.caption("Real-time telemetry and logging.")

# Read and display the log file in the UI
LOG_FILE = "agent_observability.log"

if st.sidebar.button("🔄 Refresh Logs"):
    st.rerun()

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        logs = f.readlines()
    
    with st.sidebar.expander("📝 View Raw Server Logs", expanded=False):
        # Display the last 20 lines of the log file so it doesn't clutter the UI
        recent_logs = "".join(logs[-20:])
        if recent_logs.strip():
            st.code(recent_logs, language="text")
        else:
            st.caption("Logs are currently empty.")
else:
    st.sidebar.info("No logs detected. Send a message to generate the first log entry.")

if not hf_key:
    st.warning("Please enter your Hugging Face Token in the sidebar.")
    st.stop()

headers = {"hf-key": hf_key}

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a support question..."):
    # UI-Side Latency Tracking
    ui_start_time = time.time()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                payload = {"query": prompt, "top_k": 3}
                res = requests.post("http://localhost:8000/generate", json=payload, headers=headers, timeout=20)
                
                ui_latency = time.time() - ui_start_time
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    st.markdown(answer)
                    
                    # Display Client-Side Telemetry directly under the answer
                    st.caption(f"⏱️ UI Roundtrip Time: {ui_latency:.2f}s | Check sidebar logs for detailed backend latency and token counts.")
                    
                    if data.get("sources"):
                        with st.expander("📚 Sources"):
                            for source in data["sources"]:
                                st.caption(f"📄 {source.get('filename')}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                else:
                    error_msg = res.json().get('detail', 'Unknown error.')
                    st.error(f"🚨 {error_msg}")
                    
            except requests.exceptions.Timeout:
                st.error("⏳ Connection timed out.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Could not connect to the backend server.")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {str(e)}")
