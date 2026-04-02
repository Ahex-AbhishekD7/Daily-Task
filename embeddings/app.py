import streamlit as st
import requests

st.set_page_config(page_title="Small-Doc FAISS Search", layout="wide")

st.title("📄 Local 10-Doc Vector Store")

# UPLOAD & INDEX 
st.header("1. Upload Documents")
uploaded_files = st.file_uploader("Upload 10 text files", accept_multiple_files=True, type=['txt'])

if st.button("Generate Embeddings"):
    if len(uploaded_files) == 10:
        try:
            files = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded_files]
            with st.spinner("Indexing..."):
                res = requests.post("http://localhost:8000/create-embeddings", files=files)
            
            if res.status_code == 200:
                st.success(res.json()["message"])
            else:
                st.error(res.json()["detail"])
        except Exception as e:
            st.error(f"Server Connection Error: {e}")
    else:
        st.warning(f"Need 10 files. You provided {len(uploaded_files)}.")

st.divider()

# SEARCH 
st.header("2. Semantic Search")
user_query = st.text_input("Enter your question about the documents:")

if st.button("Search"):
    if user_query:
        try:
            payload = {"query": user_query, "top_k": 1}
            res = requests.post("http://localhost:8000/search", json=payload)
            
            if res.status_code == 200:
                results = res.json()["results"]
                if results:
                    st.subheader("Top Matching Document:")
                    st.info(results[0]["text"])
                    st.caption(f"Similarity Distance: {results[0]['score']:.4f}")
            else:
                st.error(res.json()["detail"])
        except Exception as e:
            st.error(f"Search failed: {e}")
    else:
        st.warning("Please enter a query.")
