import streamlit as st
import requests

st.set_page_config(page_title="AI Support Agent", layout="centered", page_icon="🎧")
st.title("🎧 Customer Support AI")
st.caption("A resilient RAG agent with graceful degradation and failure handling.")

# Sidebar Configuration
st.sidebar.header("🔑 Configuration")
hf_key = st.sidebar.text_input("Hugging Face Token", type="password")

# Note: In a real app, the vector DB is usually initialized in the backend automatically, 
# but we kept the header requirement from the previous design for consistency.
st.sidebar.info("The agent will automatically handle API timeouts, empty retrievals, and rate limits.")

if not hf_key:
    st.warning("Please enter your Hugging Face Token in the sidebar to start chatting.")
    st.stop()

headers = {"hf-key": hf_key}

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm the support assistant. How can I help you today?"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Area
if prompt := st.chat_input("Ask a support question..."):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the Backend API
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                payload = {"query": prompt, "top_k": 3}
                # UI Timeout set slightly higher than Backend timeout (15s) to catch the 504 gracefully
                res = requests.post("http://localhost:8000/generate", json=payload, headers=headers, timeout=20)
                
                # Success
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    st.markdown(answer)
                    
                    # Display sources if it's a RAG answer (not a fallback)
                    if data.get("sources"):
                        with st.expander("📚 Knowledge Base Sources"):
                            for source in data["sources"]:
                                st.caption(f"📄 {source.get('filename', 'Unknown Document')}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Backend explicitly threw an error (502, 503, 504)
                else:
                    error_msg = res.json().get('detail', 'An unknown error occurred.')
                    st.error(f"🚨 {error_msg}")
                    
            # Connection or UI-side Timeout errors
            except requests.exceptions.Timeout:
                st.error("⏳ The connection timed out. Please try your request again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Could not connect to the backend server. Please ensure `api.py` is running.")
            except Exception as e:
                st.error(f"⚠️ Unexpected frontend error: {str(e)}")