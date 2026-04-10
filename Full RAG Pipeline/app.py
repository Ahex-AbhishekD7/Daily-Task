import streamlit as st
import requests

st.set_page_config(page_title="Optimized RAG", layout="wide")
st.title("🎯 Optimized RAG: Tuned for Accuracy")

# Sidebar Config
st.sidebar.header("🔑 Credentials")
pinecone_key = st.sidebar.text_input("Pinecone API Key", type="password")
hf_key = st.sidebar.text_input("Hugging Face Token", type="password")

st.sidebar.header("⚙️ RAG Parameters")
top_k_param = st.sidebar.slider("Top-K Retrieval Count", min_value=1, max_value=10, value=5, help="How many chunks to fetch from the DB and send to the LLM.")

if not pinecone_key or not hf_key:
    st.warning("Please enter your API Keys in the sidebar.")
    st.stop()

headers = {"pinecone-key": pinecone_key, "hf-key": hf_key}

#  DATA INGESTION 
with st.expander("📁 1. Populate Vector DB (Chunking Enabled)", expanded=False):
    st.write("Documents will be automatically split into overlapping 150-word chunks.")
    uploaded_files = st.file_uploader("Upload .txt files", accept_multiple_files=True, type=['txt'])
    
    if st.button("Chunk & Push to Pinecone"):
        if uploaded_files:
            files = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded_files]
            with st.spinner("Chunking text, creating vectors, and uploading..."):
                res = requests.post("http://localhost:8000/upsert", files=files, headers=headers)
                if res.status_code == 200:
                    st.success(res.json()["message"])
                else:
                    st.error(f"Error: {res.json()['detail']}")
        else:
            st.warning("Please upload files first.")

st.divider()

#  QUERY & EVALUATE
st.header("💬 2. Ask the Data")
query = st.text_input("Enter your precise question:")

if st.button("Generate Verified Answer"):
    if query:
        with st.spinner("Retrieving Top-K chunks and generating response..."):
            payload = {"query": query, "top_k": top_k_param}
            res = requests.post("http://localhost:8000/generate", json=payload, headers=headers)
            
            if res.status_code == 200:
                data = res.json()
                
                # Display Answer
                st.subheader("🤖 LLM Answer")
                st.info(data["answer"])
                
                # Display Sources for Accuracy Evaluation
                st.subheader("🔍 Retrieval Evaluation (Context Provided to LLM)")
                if data["sources"]:
                    cols = st.columns(min(3, len(data["sources"])))
                    for i, source in enumerate(data["sources"]):
                        col = cols[i % 3]
                        with col:
                            st.markdown(f"""
                            **{source['filename']}** `Chunk: {source['chunk']}`  
                            `Similarity: {source['score']:.3f}`
                            """)
                else:
                    st.warning("No context was retrieved.")
            else:
                st.error(f"🚨 Generation Failed: {res.json().get('detail', 'Unknown error')}")
