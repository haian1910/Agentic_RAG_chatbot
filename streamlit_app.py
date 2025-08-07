import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Configure page
st.set_page_config(
    page_title="QA RAG System",
    page_icon="🤖",
    layout="wide"
)

# API endpoint
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.title("🤖 QA RAG System")
st.markdown("Upload PDF documents and ask questions about them!")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload a PDF document to index it in the vector store"
    )
    
    if uploaded_file is not None:
        if st.button("Upload & Index"):
            with st.spinner("Uploading and indexing document..."):
                try:
                    files = {"file": uploaded_file}
                    response = requests.post(f"{API_BASE}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Main chat interface
st.header("💬 Ask Questions")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE}/query",
                    json={"question": prompt}
                )
                
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to API. Make sure the FastAPI server is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Information section
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Upload Documents**: Use the sidebar to upload PDF documents
    2. **Ask Questions**: Type questions in the chat input
    3. **Get Answers**: The system will search your documents first, then the web if needed
    
    **Example questions:**
    - What are the key points in the document?
    - What is Tesla's Q3 2024 revenue?
    - Latest news about artificial intelligence?
    """)

# Status check
with st.expander("🔧 System Status"):
    if st.button("Check API Status"):
        try:
            response = requests.get(f"{API_BASE}/health")
            if response.status_code == 200:
                st.success("✅ API is healthy")
            else:
                st.error("❌ API is not responding properly")
        except:
            st.error("❌ Cannot connect to API")