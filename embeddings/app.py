import streamlit as st
import requests

st.set_page_config(page_title="AI Embedding Creator")
st.title("📄 Document Embedding Creator")
st.write("Upload exactly 10 text files to create and store local embeddings.")

uploaded_files = st.file_uploader("Choose 10 .txt files", accept_multiple_files=True, type=['txt'])

if st.button("Generate & Store"):
    if len(uploaded_files) == 10:
        try:
            # Prepare files for the API request
            files = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded_files]
            
            with st.spinner("Processing embeddings..."):
                response = requests.post("http://localhost:8000/create-embeddings", files=files)
            
            if response.status_code == 200:
                st.success(response.json().get("message"))
            else:
                st.error(f"Error: {response.json().get('detail')}")
                
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
    else:
        st.warning(f"You've uploaded {len(uploaded_files)} files. Please upload exactly 10.")