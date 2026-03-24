import streamlit as st
import requests
import io

# Constants
API_URL = "http://localhost:8000/process-email"
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

st.set_page_config(page_title="AI Email Agent", layout="centered")

st.title("📧 AI Email Processing Agent")
st.write("Analyze and draft responses for your emails using Llama 3.1.")

# Input options
input_method = st.radio("Choose input method:", ("Paste Text", "Upload .txt File"))

email_content = ""

if input_method == "Paste Text":
    email_content = st.text_area("Paste your email content here:", height=250)

elif input_method == "Upload .txt File":
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    
    if uploaded_file is not None:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error(f"File size exceeds the {MAX_FILE_SIZE_MB}MB limit. Please upload a smaller file.")
        else:
            # Read file content
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            email_content = stringio.read()
            st.success("File uploaded successfully!")
            with st.expander("View Uploaded Content"):
                st.write(email_content)

# Submit button
if st.button("Process Email", type="primary"):
    if not email_content.strip():
        st.warning("Please provide email content to process.")
    else:
        with st.spinner("Agent is analyzing and drafting..."):
            try:
                response = requests.post(
                    API_URL, 
                    json={"email_text": email_content},
                    timeout=60 # Extended timeout for LLM processing
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()
                    st.subheader("📊 Summary")
                    st.info(result.get("summary"))
                    
                    st.subheader("🎯 Intent")
                    st.success(result.get("intent"))
                    
                    st.subheader("✍️ Draft Response")
                    st.write(result.get("draft"))
                    
                else:
                    st.error(f"Error from API: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"An error occurred: {e}")