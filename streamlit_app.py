import streamlit as st
import requests
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

# Configure page
st.set_page_config(
    page_title="QA RAG System with Memory",
    page_icon="🤖",
    layout="wide"
)

# API endpoint
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def create_new_session():
    """Create a new chat session"""
    try:
        response = requests.post(f"{API_BASE}/session/create")
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.messages = []
            st.success(f"✅ New session created: {st.session_state.session_id[:8]}...")
        else:
            st.error("❌ Failed to create new session")
    except Exception as e:
        st.error(f"❌ Error creating session: {str(e)}")

def clear_current_session():
    """Clear the current session memory"""
    if st.session_state.session_id:
        try:
            response = requests.post(
                f"{API_BASE}/session/clear",
                json={"session_id": st.session_state.session_id}
            )
            if response.status_code == 200:
                st.session_state.messages = []
                st.success("✅ Session memory cleared")
            else:
                st.error("❌ Failed to clear session")
        except Exception as e:
            st.error(f"❌ Error clearing session: {str(e)}")

def load_chat_history():
    """Load chat history from the backend"""
    if st.session_state.session_id:
        try:
            response = requests.get(f"{API_BASE}/session/{st.session_state.session_id}/history")
            if response.status_code == 200:
                data = response.json()
                st.session_state.messages = data["messages"]
                st.success("✅ Chat history loaded")
            elif response.status_code == 404:
                # Session doesn't exist, create new one
                create_new_session()
            else:
                st.error("❌ Failed to load chat history")
        except Exception as e:
            st.error(f"❌ Error loading history: {str(e)}")

st.title("🤖 QA RAG System with Memory")
st.markdown("Upload PDF documents and have conversations with memory!")

# Sidebar for session management and document upload
with st.sidebar:
    st.header("🔧 Session Management")
    
    # Display current session
    if st.session_state.session_id:
        st.info(f"Current Session: {st.session_state.session_id[:8]}...")
    else:
        st.warning("No active session")
    
    # Session controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Session"):
            create_new_session()
    
    with col2:
        if st.button("🧹 Clear Chat"):
            clear_current_session()
    
    if st.button("📥 Load History"):
        load_chat_history()
    
    st.divider()
    
    # Document upload section
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
                    data = {}
                    if st.session_state.session_id:
                        data["session_id"] = st.session_state.session_id
                    
                    response = requests.post(f"{API_BASE}/upload", files=files, data=data)
                    
                    if response.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Create session if none exists
if st.session_state.session_id is None:
    create_new_session()

# Main chat interface
st.header("💬 Chat with Memory")

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
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Update session ID if it was created by the backend
                    if "session_id" in data:
                        st.session_state.session_id = data["session_id"]
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

# Information section
with st.expander("ℹ️ How to use"):
    st.markdown("""
    **New Features with Memory:**
    - 🧠 **Memory**: The system now remembers your conversation history
    - 💬 **Context**: You can refer to previous questions and answers
    - 🔄 **Sessions**: Each session maintains its own conversation history
    - 🧹 **Clear**: Clear the conversation memory when needed
    
    **How to use:**
    1. **Start a Session**: A new session is created automatically
    2. **Upload Documents**: Use the sidebar to upload PDF documents
    3. **Have Conversations**: Ask questions and refer to previous discussions
    4. **Manage Memory**: Create new sessions or clear current conversation
    
    **Example conversations:**
    - "What are the key points in the document?"
    - "Can you elaborate on the second point you mentioned?"
    - "How does this relate to what we discussed earlier?"
    - "Summarize our conversation so far"
    """)

# Status and statistics
with st.expander("📊 Session Statistics"):
    if st.button("Check System Status"):
        try:
            response = requests.get(f"{API_BASE}/health")
            if response.status_code == 200:
                data = response.json()
                st.success("✅ API is healthy")
                st.info(f"Active sessions: {data.get('active_sessions', 0)}")
                st.info(f"Vector store available: {data.get('vectorstore_available', False)}")
            else:
                st.error("❌ API is not responding properly")
        except:
            st.error("❌ Cannot connect to API")
    
    # Show current conversation stats
    if st.session_state.messages:
        user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        assistant_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
        st.metric("Messages in this session", user_messages + assistant_messages)
        st.metric("Questions asked", user_messages)
        st.metric("Responses received", assistant_messages)