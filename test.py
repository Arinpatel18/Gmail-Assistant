import os
import base64
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_core.documents import Document

# ---------------------------
# CONFIGURATION
# ---------------------------
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MAX_EMAILS = 100  # number of recent emails from Inbox

# ---------------------------
# AUTHENTICATION
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

# Fetch top 50 emails from Inbox
results = (
    service.users()
    .messages()
    .list(userId="me", labelIds=["INBOX"], maxResults=MAX_EMAILS)
    .execute()
)
messages = results.get("messages", [])
print(f"📩 Found {len(messages)} inbox messages")

# ---------------------------
# PARSE EMAILS INTO DOCUMENTS
# ---------------------------
documents = []

for i, msg in enumerate(messages, start=1):
    msg_data = (
        service.users()
        .messages()
        .get(userId="me", id=msg["id"], format="full")
        .execute()
    )

    payload = msg_data.get("payload", {})
    headers = payload.get("headers", [])

    subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "")
    sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "")
    date = next((h["value"] for h in headers if h["name"].lower() == "date"), "")

    # Extract plain text body
    body = ""
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part["body"].get("data")
                if data:
                    body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    break
    else:
        data = payload.get("body", {}).get("data")
        if data:
            body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    # Construct LangChain Document
    doc = Document(
        page_content=body.strip(),
        metadata={
            "subject": subject,
            "from": sender,
            "date": date,
            "id": msg["id"],
        },
    )
    documents.append(doc)

    print(f"✅ Loaded email {i}: {subject[:60]}...")

# ---------------------------
# SUMMARY
# ---------------------------
print("\n🎉 Successfully loaded", len(documents), "emails as Documents.")

# Optional: display a sample
for i, doc in enumerate(documents[:3], start=1):
    print(f"\n--- Email {i} ---")
    print("From:", doc.metadata["from"])
    print("Subject:", doc.metadata["subject"])
    print("Date:", doc.metadata["date"])
    print("Body preview:", doc.page_content[:200].replace("\n", " "))
    print("=" * 60)
