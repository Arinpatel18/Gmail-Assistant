import os
import base64
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment. Please add it to your .env file.")

# ---------------------------
# CONFIGURATION
# ---------------------------
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MAX_EMAILS = 100
DB_PATH = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10

# ---------------------------
# AUTHENTICATE GMAIL
# ---------------------------
creds = None
if os.path.exists(TOKEN_FILE):
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
    with open(TOKEN_FILE, "w") as token:
        token.write(creds.to_json())

# ---------------------------
# CONNECT TO GMAIL API
# ---------------------------
service = build("gmail", "v1", credentials=creds)

# Fetch ALL emails without any filter
results = (
    service.users()
    .messages()
    .list(userId="me", maxResults=MAX_EMAILS)
    .execute()
)
messages = results.get("messages", [])
print(f"📩 Found {len(messages)} total messages")
print(f"{'='*100}\n")

# ---------------------------
# HELPER FUNCTION TO CLEAN HTML
# ---------------------------
def clean_html(html_content):
    """Convert HTML to clean text"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        # Get text and clean up whitespace
        text = soup.get_text(separator=' ')
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        return html_content

# ---------------------------
# HELPER FUNCTION TO EXTRACT EMAIL BODY (COMPREHENSIVE)
# ---------------------------
def get_email_body(payload):
    """Extract email body - handles ALL formats"""
    body_parts = []
    
    def decode_data(data):
        """Safely decode base64 data"""
        if not data:
            return ""
        try:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"    ⚠️ Decode error: {e}")
            return ""
    
    def extract_parts(parts, level=0):
        """Recursively extract all text content"""
        for part in parts:
            mime_type = part.get("mimeType", "")
            
            # Handle nested parts
            if "parts" in part:
                extract_parts(part["parts"], level + 1)
            
            # Extract any text content
            elif mime_type.startswith("text/"):
                body_data = part.get("body", {})
                data = body_data.get("data")
                
                if data:
                    decoded = decode_data(data)
                    if decoded.strip():
                        if mime_type == "text/html":
                            decoded = clean_html(decoded)
                        body_parts.append(decoded)
    
    # Try to extract from parts
    if "parts" in payload:
        extract_parts(payload["parts"])
    
    # Try direct body
    if not body_parts:
        body_data = payload.get("body", {})
        data = body_data.get("data")
        if data:
            decoded = decode_data(data)
            if decoded.strip():
                mime_type = payload.get("mimeType", "")
                if "html" in mime_type.lower():
                    decoded = clean_html(decoded)
                body_parts.append(decoded)
    
    # Try snippet as last resort
    if not body_parts:
        return ""
    
    # Combine all parts
    full_body = "\n\n".join(body_parts)
    return full_body.strip()

# ---------------------------
# PARSE EMAILS INTO DOCUMENTS
# ---------------------------
print("📋 FETCHING EMAILS...")
print(f"{'='*100}\n")

documents = []
skipped_count = 0

for idx, msg in enumerate(messages, 1):
    try:
        msg_data = (
            service.users()
            .messages()
            .get(userId="me", id=msg["id"], format="full")
            .execute()
        )
        payload = msg_data.get("payload", {})
        headers = payload.get("headers", [])
        snippet = msg_data.get("snippet", "")

        # Extract headers
        subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "No Subject")
        sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown")
        date = next((h["value"] for h in headers if h["name"].lower() == "date"), "Unknown Date")
        labels = msg_data.get("labelIds", [])

        # Extract body
        body = get_email_body(payload)
        
        # Use snippet if body is empty
        if not body.strip() and snippet:
            body = snippet

        # Only add to documents if there's actual content
        if body.strip():
            # Truncate very long emails
            max_body_length = 4000
            if len(body) > max_body_length:
                body = body[:max_body_length] + "... [truncated]"

            doc = Document(
                page_content=body.strip(),
                metadata={
                    "subject": subject,
                    "from": sender,
                    "date": date,
                    "id": msg["id"],
                    "labels": ", ".join(labels),
                    "email_number": idx
                },
            )
            documents.append(doc)
        else:
            skipped_count += 1
        
    except Exception as e:
        print(f"❌ EMAIL #{idx}: Error - {e}")
        skipped_count += 1
        continue

print(f"\n{'='*100}")
print(f"✅ Successfully loaded {len(documents)} emails with content")
print(f"⚠️ Skipped {skipped_count} emails (empty or errors)")
print(f"{'='*100}\n")

if len(documents) == 0:
    print("❌ No emails with content found!")
    print("\nDEBUGGING TIPS:")
    print("1. Check if your emails are in a different format")
    print("2. Try accessing Gmail web interface to verify emails exist")
    print("3. Check OAuth permissions - you need 'gmail.readonly' scope")
    exit(1)

# ---------------------------
# BUILD OR LOAD VECTOR STORE
# ---------------------------
print("🔧 Building vector store...")
embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Check if we should rebuild
rebuild = True
if os.path.exists(DB_PATH):
    user_input = input("📦 Existing FAISS index found. Rebuild? (y/n): ").lower()
    rebuild = user_input == 'y'

if rebuild:
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)
    print("🔄 Building new FAISS index...")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(DB_PATH)
    print(f"💾 Created and saved FAISS index at {DB_PATH}")
else:
    vectorstore = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    print(f"📦 Loaded existing FAISS index from {DB_PATH}")

print(f"\n{'='*100}")

# ---------------------------
# INITIALIZE LLM
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
    google_api_key=api_key
)

# ---------------------------
# QUERY LOOP - ASK MULTIPLE QUESTIONS
# ---------------------------
print("\n🤖 Gmail Assistant Ready!")
print("Type your questions about your emails. Type 'quit' or 'exit' to stop.\n")

while True:
    print(f"{'='*100}")
    query = input("\n🧠 Enter your query (or 'quit' to exit): ").strip()
    
    if query.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    if not query:
        print("⚠️ Please enter a valid query.")
        continue
    
    # Retrieve relevant emails
    results = vectorstore.similarity_search(query, k=min(TOP_K, len(documents)))
    
    print(f"\n{'='*100}")
    print(f"📧 RETRIEVED TOP {len(results)} RELEVANT EMAILS FOR YOUR QUERY")
    print(f"{'='*100}\n")
    
    # Print retrieved emails
    for i, doc in enumerate(results, 1):
        print(f"--- RELEVANT EMAIL {i} ---")
        print(f"Subject: {doc.metadata.get('subject', 'No Subject')}")
        print(f"From: {doc.metadata.get('from', 'Unknown')}")
        print(f"Date: {doc.metadata.get('date', 'Unknown Date')}")
        body_preview = doc.page_content[:300].replace('\n', ' ')
        print(f"Body Preview: {body_preview}...")
        print(f"{'-'*100}\n")
    
    # Build context for LLM
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
5. If the query asks about specific emails, reference them by their subject line
6. If none of the emails are relevant, say so clearly

Please provide your response:"""
    
    print(f"{'='*100}")
    print("🤖 GENERATING ANSWER USING GEMINI...")
    print(f"{'='*100}\n")
    
    try:
        response = llm.invoke(prompt)
        print("💡 GEMINI RESPONSE:\n")
        print(response.content)
        print(f"\n{'='*100}\n")
    except Exception as e:
        print(f"❌ Error generating response: {e}\n")