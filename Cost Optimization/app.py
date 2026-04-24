import streamlit as st
import requests
import time

st.set_page_config(page_title="Cost-Optimized AI Agent", layout="wide", page_icon="💰")
st.title("💰 Customer Support AI (Cost-Optimized)")
st.caption("RAG Pipeline with strict token constraints and context thresholding.")

# SIDEBAR
st.sidebar.header("🔑 Configuration")
hf_key = st.sidebar.text_input("Hugging Face Token", type="password")

if not hf_key:
    st.warning("Please enter your Hugging Face Token.")
    st.stop()

headers = {"hf-key": hf_key}

# MAIN CHAT
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a support question..."):
    ui_start = time.time()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing (Optimized)..."):
            try:
                res = requests.post("http://localhost:8000/generate", json={"query": prompt}, headers=headers, timeout=15)
                ui_latency = time.time() - ui_start
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    st.markdown(answer)
                    
                    # Display explicit token savings and metrics
                    tokens = data.get("telemetry", {})
                    p_tokens = tokens.get("prompt_tokens", "N/A")
                    c_tokens = tokens.get("completion_tokens", "N/A")
                    
                    st.caption(f"⏱️ {ui_latency:.2f}s | 🪙 Tokens used - Prompt: {p_tokens}, Completion: {c_tokens}")
                    
                    if data.get("sources"):
                        with st.expander("📚 Filtered Sources (Score > 0.5)"):
                            for source in data["sources"]:
                                st.caption(f"📄 {source.get('filename')} (Score: {source.get('score', 0):.2f})")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"🚨 {res.json().get('detail', 'Error')}")
                    
            except requests.exceptions.Timeout:
                st.error("⏳ Request timed out.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to backend.")