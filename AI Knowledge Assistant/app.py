# app.py
import streamlit as st
import requests

st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
st.title("🧠 AI Knowledge Assistant")

# SIDEBAR: CONFIG & UPLOAD 
st.sidebar.header("🔑 Setup")
hf_key = st.sidebar.text_input("Hugging Face Token", type="password")

st.sidebar.divider()

st.sidebar.header("📄 Document Upload")
uploaded_files = st.sidebar.file_uploader("Upload Knowledge (.txt)", accept_multiple_files=True, type=['txt'])

if st.sidebar.button("Index Documents"):
    if not uploaded_files:
        st.sidebar.warning("Upload files first.")
    else:
        with st.sidebar.status("Processing documents..."):
            files = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded_files]
            try:
                res = requests.post("http://localhost:8000/upload", files=files)
                if res.status_code == 200:
                    st.sidebar.success(res.json()["message"])
                else:
                    st.sidebar.error(f"Error: {res.json().get('detail')}")
            except requests.exceptions.ConnectionError:
                st.sidebar.error("🔌 Could not connect to API.")

if not hf_key:
    st.warning("Please enter your Hugging Face Token in the sidebar.")
    st.stop()

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Documents uploaded? Ask me anything."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            try:
                headers = {"hf-key": hf_key}
                res = requests.post("http://localhost:8000/ask", json={"query": prompt}, headers=headers, timeout=20)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    st.markdown(answer)
                    
                    if data.get("sources"):
                        st.caption(f"📚 Sources: {', '.join(data['sources'])}")
                        
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"🚨 {res.json().get('detail', 'Error')}")
                    
            except requests.exceptions.Timeout:
                st.error("⏳ Request timed out.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to backend.")