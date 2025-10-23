import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Gmail AI Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
    }
    h2 {
        color: #374151;
        font-weight: 600;
    }
    h3 {
        color: #4b5563;
        font-weight: 500;
    }
    .email-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .response-box {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0ea5e9;
        margin-top: 1rem;
        color: #1f2937;
        font-size: 16px;
        line-height: 1.6;
    }
    .response-box p {
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .response-box strong {
        color: #111827;
    }
    .response-box a {
        color: #2563eb;
    }
    .sidebar .sidebar-content {
        background-color: #1f2937;
    }
    </style>
""", unsafe_allow_html=True)


# LOAD ENVIRONMENT VARIABLES

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


# CONFIGURATION

DB_PATH = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10


# INITIALIZE SESSION STATE

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'total_emails' not in st.session_state:
    st.session_state.total_emails = 0


# LOAD MODELS

@st.cache_resource
def load_vectorstore():
    """Load FAISS vector store"""
    if not os.path.exists(DB_PATH):
        return None, 0
    
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    
    # Get number of documents
    total_docs = vectorstore.index.ntotal
    
    return vectorstore, total_docs

@st.cache_resource
def load_llm():
    """Load Gemini LLM"""
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.5,
        max_tokens=2048,
        google_api_key=api_key
    )


# HEADER

st.title("📧 Gmail AI Assistant")
st.markdown("### Your intelligent email companion powered by AI")
st.markdown("---")


# SIDEBAR

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/gmail.png", width=80)
    st.title("⚙️ Settings")
    
    st.markdown("### 📊 System Status")
    
    # Check vector store
    if os.path.exists(DB_PATH):
        st.success("✅ Vector Store: Ready")
        if st.session_state.vectorstore is None:
            with st.spinner("Loading vector store..."):
                st.session_state.vectorstore, st.session_state.total_emails = load_vectorstore()
        st.info(f"📨 Emails Indexed: **{st.session_state.total_emails}**")
    else:
        st.error("❌ Vector Store: Not Found")
        st.warning("⚠️ Please run `python data_preprocess.py` first!")
    
    # Check LLM
    if api_key:
        st.success("✅ Gemini API: Connected")
        if st.session_state.llm is None:
            st.session_state.llm = load_llm()
    else:
        st.error("❌ Gemini API: Missing")
        st.warning("⚠️ Add GOOGLE_API_KEY to .env file")
    
    st.markdown("---")
    
    # Configuration
    st.markdown("### 🎛️ Query Settings")
    top_k = st.slider("Number of emails to retrieve", 3, 15, TOP_K)
    
    st.markdown("---")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **First Time Setup:**
           - Run `python data_preprocess.py`
           - This fetches and indexes your emails
        
        2. **Ask Questions:**
           - Type your query in the search box
           - Get AI-powered answers from your emails
        
        3. **View Results:**
           - See relevant emails and AI response
           - All information comes from your Gmail
        """)
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# MAIN CONTENT


# Check if system is ready
if not os.path.exists(DB_PATH):
    st.error("🚫 Vector store not found!")
    st.info("👉 Please run the following command first:")
    st.code("python data_preprocess.py", language="bash")
    st.stop()

if not api_key:
    st.error("🚫 Google API Key not found!")
    st.info("👉 Please add GOOGLE_API_KEY to your .env file")
    st.stop()

# Statistics cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin:0; color:white;">📨</h2>
        <h3 style="margin:0.5rem 0; color:white;">{st.session_state.total_emails}</h3>
        <p style="margin:0; color:rgba(255,255,255,0.9);">Emails Indexed</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin:0; color:white;">🤖</h2>
        <h3 style="margin:0.5rem 0; color:white;">{len(st.session_state.chat_history)}</h3>
        <p style="margin:0; color:rgba(255,255,255,0.9);">Queries Processed</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin:0; color:white;">📊</h2>
        <h3 style="margin:0.5rem 0; color:white;">{top_k}</h3>
        <p style="margin:0; color:rgba(255,255,255,0.9);">Top Results</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Query input
query = st.text_input(
    "🔍 Ask me anything about your emails:",
    placeholder="e.g., What emails did I receive about internships?",
    key="query_input"
)

search_button = st.button("🚀 Search", type="primary", use_container_width=True)

# Process query
if search_button and query:
    if st.session_state.vectorstore is None or st.session_state.llm is None:
        st.error("System not initialized properly. Please refresh the page.")
        st.stop()
    
    with st.spinner("🔍 Searching through your emails..."):
        # Retrieve relevant emails
        results = st.session_state.vectorstore.similarity_search(query, k=top_k)
        
        st.markdown("---")
        st.markdown(f"## 📧 Top {len(results)} Relevant Emails")
        
        # Display relevant emails
        for i, doc in enumerate(results, 1):
            with st.expander(f"📨 Email {i}: {doc.metadata.get('subject', 'No Subject')}", expanded=(i <= 3)):
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.markdown(f"**From:** {doc.metadata.get('from', 'Unknown')}")
                    st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown')}")
                
                with col_b:
                    st.markdown(f"**Labels:** {doc.metadata.get('labels', 'None')}")
                
                st.markdown("**Content:**")
                body_preview = doc.page_content[:500]
                st.text_area(
                    "Email Body",
                    body_preview + ("..." if len(doc.page_content) > 500 else ""),
                    height=150,
                    key=f"email_{i}",
                    label_visibility="collapsed"
                )
    
    with st.spinner("🤖 Generating AI response..."):
        # Build context
        context_text = "\n\n".join([
            f"EMAIL {i+1}:\n"
            f"Subject: {doc.metadata.get('subject', 'No Subject')}\n"
            f"From: {doc.metadata.get('from', 'Unknown')}\n"
            f"Date: {doc.metadata.get('date', 'Unknown Date')}\n"
            f"Content:\n{doc.page_content}\n"
            f"{'-'*40}"
            for i, doc in enumerate(results)
        ])
        
        # Generate response
        prompt = f"""You are an intelligent email assistant. Analyze the following {len(results)} Gmail messages that are most relevant to the user's query.

USER QUERY:
{query}

RELEVANT EMAILS:
{context_text}

INSTRUCTIONS:
1. Carefully read through all the emails provided above
2. Identify which emails are relevant to answering the user's query
3. Provide a clear, concise, and helpful answer based on the email content
4. If the emails contain specific information (dates, names, amounts, links, deadlines, etc.), include those details
5. Reference emails by their subject line when mentioning them
6. If none of the emails are directly relevant, explain what you found instead
7. Always provide a complete answer - do not just extract single pieces of information
8. Format your response in a friendly, conversational manner with proper paragraphs

IMPORTANT: Provide a detailed response with complete sentences and explanations. Do not just extract email addresses or single facts.

Your response:"""
        
        try:
            response = st.session_state.llm.invoke(prompt)
            
            # Debug: Check response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Display AI response
            st.markdown("---")
            st.markdown("## 💡 AI Assistant Response")
            
            if response_text and len(response_text.strip()) > 0:
                st.markdown(f"""
                <div class="response-box">
                    {response_text}
                </div>
                """, unsafe_allow_html=True)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": query,
                    "response": response_text,
                    "email_count": len(results)
                })
            else:
                st.warning("⚠️ The AI returned an empty response. Please try rephrasing your query.")
                st.info(f"Debug info: Response type: {type(response)}, Content: {response}")
            
        except Exception as e:
            st.error(f"❌ Error generating response: {e}")
            st.error(f"Full error details: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 📜 Recent Queries")
    
    for idx, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
        with st.expander(f"🔍 Query {len(st.session_state.chat_history) - idx + 1}: {chat['query'][:50]}..."):
            st.markdown(f"**Query:** {chat['query']}")
            st.markdown(f"**Emails Found:** {chat['email_count']}")
            st.markdown("**Response:**")
            st.info(chat['response'])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>🔒 Your emails are processed locally and securely</p>
    <p>Made with ❤️ using Streamlit, LangChain & Google Gemini</p>
</div>
""", unsafe_allow_html=True)
