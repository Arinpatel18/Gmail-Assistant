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


load_dotenv()

# CONFIGURATION

CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MAX_EMAILS = 100
DB_PATH = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"


# HELPER FUNCTION TO CLEAN HTML

def clean_html(html_content):
    """Convert HTML to clean text"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        return html_content


# HELPER FUNCTION TO EXTRACT EMAIL BODY

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
            return ""
    
    def extract_parts(parts, level=0):
        """Recursively extract all text content"""
        for part in parts:
            mime_type = part.get("mimeType", "")
            
            if "parts" in part:
                extract_parts(part["parts"], level + 1)
            elif mime_type.startswith("text/"):
                body_data = part.get("body", {})
                data = body_data.get("data")
                
                if data:
                    decoded = decode_data(data)
                    if decoded.strip():
                        if mime_type == "text/html":
                            decoded = clean_html(decoded)
                        body_parts.append(decoded)
    
    if "parts" in payload:
        extract_parts(payload["parts"])
    
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
    
    full_body = "\n\n".join(body_parts)
    return full_body.strip()


# AUTHENTICATE GMAIL

def authenticate_gmail():
    """Authenticate and return Gmail service"""
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
    
    return build("gmail", "v1", credentials=creds)


# FETCH AND PROCESS EMAILS

def fetch_and_process_emails(max_emails=MAX_EMAILS):
    """Fetch emails and return documents"""
    print(f"{'='*100}")
    print("🔐 Authenticating with Gmail...")
    service = authenticate_gmail()
    print("✅ Authentication successful!\n")
    
    print(f"📩 Fetching {max_emails} recent emails...")
    results = service.users().messages().list(userId="me", maxResults=max_emails).execute()
    messages = results.get("messages", [])
    print(f"✅ Found {len(messages)} total messages\n")
    
    print("📋 Processing emails...")
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

            subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "No Subject")
            sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown")
            date = next((h["value"] for h in headers if h["name"].lower() == "date"), "Unknown Date")
            labels = msg_data.get("labelIds", [])

            body = get_email_body(payload)
            
            if not body.strip() and snippet:
                body = snippet

            if body.strip():
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
            
            if idx % 10 == 0:
                print(f"   Processed {idx}/{len(messages)} emails...")
            
        except Exception as e:
            print(f"   ⚠️ Error processing email #{idx}: {e}")
            skipped_count += 1
            continue

    print(f"\n✅ Successfully processed {len(documents)} emails")
    print(f"⚠️ Skipped {skipped_count} emails (empty or errors)\n")
    
    return documents


# CREATE VECTOR STORE

def create_vector_store(documents, db_path=DB_PATH):
    """Create and save FAISS vector store"""
    print("🧠 Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    print("💾 Building FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
    
    vectorstore.save_local(db_path)
    print(f"✅ Vector store saved at: {db_path}\n")
    
    return vectorstore


# MAIN EXECUTION

def main():
    print(f"\n{'='*100}")
    print("📧 GMAIL DATA PREPROCESSING & EMBEDDING CREATION")
    print(f"{'='*100}\n")
    
    # Check if vector store already exists
    if os.path.exists(DB_PATH):
        rebuild = input(f"📦 Vector store already exists at '{DB_PATH}'. Rebuild? (y/n): ").lower()
        if rebuild != 'y':
            print("\n✅ Using existing vector store. Exiting...")
            return
    
    # Fetch and process emails
    documents = fetch_and_process_emails(MAX_EMAILS)
    
    if len(documents) == 0:
        print("❌ No emails with content found!")
        print("\nDEBUGGING TIPS:")
        print("1. Check if your emails are in a different format")
        print("2. Try accessing Gmail web interface to verify emails exist")
        print("3. Check OAuth permissions - you need 'gmail.readonly' scope")
        return
    
    # Create vector store
    create_vector_store(documents)
    
    print(f"{'='*100}")
    print("🎉 DATA PREPROCESSING COMPLETE!")
    print(f"{'='*100}")
    print(f"\n📊 Summary:")
    print(f"   • Emails processed: {len(documents)}")
    print(f"   • Vector store location: {DB_PATH}")
    print(f"   • Embedding model: {MODEL_NAME}")
    print(f"\n✅ You can now run 'streamlit run app.py' to use the Gmail Assistant!\n")

if __name__ == "__main__":
    main()
