import streamlit as st
import requests
import io

API_URL = "http://localhost:8000/process-email"
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

st.set_page_config(page_title="AI Email Agent", layout="centered")

st.title("📧 AI Email Processing Agent")
st.write("Analyze and draft responses for your emails using Llama 3.1 and Pinecone Memory.")

input_method = st.radio("Choose input method:", ("Paste Text", "Upload .txt File"))
email_content = ""

if input_method == "Paste Text":
    email_content = st.text_area("Paste your email content here:", height=250)
elif input_method == "Upload .txt File":
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error(f"File size exceeds the {MAX_FILE_SIZE_MB}MB limit.")
        else:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            email_content = stringio.read()
            st.success("File uploaded successfully!")
            with st.expander("View Uploaded Content"):
                st.write(email_content)

if st.button("Process Email", type="primary"):
    if not email_content.strip():
        st.warning("Please provide email content to process.")
    else:
        with st.spinner("Agent is analyzing, searching memory, and drafting..."):
            try:
                response = requests.post(
                    API_URL, 
                    json={"user_id": "default_user", "email_text": email_content},
                    timeout=90 
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.divider()
                    
                    # --- METRICS ROW ---
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🎯 Intent")
                        st.info(result.get("intent"))
                    with col2:
                        st.subheader("🚨 Urgency")
                        urgency_val = result.get("urgency", "").lower()
                        if "high" in urgency_val:
                            st.error(f"**{result.get('urgency')}**")
                        elif "low" in urgency_val:
                            st.success(f"**{result.get('urgency')}**")
                        else:
                            st.warning(f"**{result.get('urgency')}**")

                    # --- SUMMARY & CONTEXT ---
                    st.subheader("📊 Summary")
                    st.write(result.get("summary"))
                    
                    past_context = result.get("past_context_used", "")
                    if past_context and past_context != "No relevant past emails found.":
                        with st.expander("🧠 Click to view past emails used for this draft"):
                            st.write(past_context)

                    st.divider()

                    # --- REPLIES ---
                    st.subheader("💡 Quick Reply Suggestions")
                    st.write(result.get("smart_replies"))
                    
                    st.subheader("✍️ Full Draft Response")
                    st.text_area("You can edit this draft before sending:", value=result.get("draft"), height=250)
                    
                else:
                    st.error(f"Error from API: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"An error occurred: {e}")
