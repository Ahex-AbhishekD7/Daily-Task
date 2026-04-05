import streamlit as st
import requests

st.set_page_config(page_title="RAG Evaluation Pipeline", layout="wide")
st.title("☁️ Pinecone Retrieval & Evaluation Pipeline")

# Sidebar for API Key
st.sidebar.header("Configuration")
pinecone_key = st.sidebar.text_input("Pinecone API Key", type="password")

if not pinecone_key:
    st.warning("Please enter your Pinecone API Key in the sidebar to continue.")
    st.stop()

headers = {"pinecone-key": pinecone_key}

# UPSERT
st.header("1. Populate Vector Database")
uploaded_files = st.file_uploader("Upload Text Documents", accept_multiple_files=True, type=['txt'])

if st.button("Push to Pinecone"):
    if uploaded_files:
        files = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded_files]
        with st.spinner("Embedding and uploading to Pinecone..."):
            res = requests.post("http://localhost:8000/upsert", files=files, headers=headers)
            if res.status_code == 200:
                st.success(res.json()["message"])
            else:
                st.error(f"Error: {res.json()['detail']}")
    else:
        st.warning("Please upload files first.")

st.divider()

# RETRIEVE & EVALUATE 
st.header("2. Retrieve & Check Relevance")
query = st.text_input("Enter your query:")
top_k = st.slider("Top K Results to Retrieve", min_value=1, max_value=10, value=3)

if st.button("Search"):
    if query:
        with st.spinner("Searching Pinecone and evaluating relevance..."):
            payload = {"query": query, "top_k": top_k}
            res = requests.post("http://localhost:8000/retrieve", json=payload, headers=headers)
            
            if res.status_code == 200:
                data = res.json()
                st.subheader("Evaluated Results")
                
                for i, doc in enumerate(data["results"]):
                    with st.expander(f"Result {i+1}: {doc['filename']} | Relevance: {doc['relevance_score']:.2f}"):
                        st.markdown(f"**Pinecone Cosine Similarity:** `{doc['pinecone_similarity']:.4f}`")
                        st.markdown(f"**Cross-Encoder Relevance:** `{doc['relevance_score']:.4f}`")
                        st.write("---")
                        st.write(doc['text'])
                        
                        # Add a visual flag for bad retrievals
                        if doc['relevance_score'] < 0: 
                            st.error("⚠️ Pipeline Evaluation Warning: This document is likely irrelevant to the specific query.")
                        else:
                            st.success("✅ Document evaluated as relevant.")
            else:
                st.error(f"Search failed: {res.json()['detail']}")