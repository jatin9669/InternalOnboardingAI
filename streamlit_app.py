import streamlit as st
import os
import pickle
import fitz  # PyMuPDF for PDF processing
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import re
import gdown
import zipfile
from urllib.parse import urlparse
import requests
from pathlib import Path
import json
import io
import hashlib
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
import streamlit_authenticator as stauth
from datetime import datetime, timedelta

# Allow insecure transport for local development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'credentials' not in st.session_state:
    st.session_state.credentials = None
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""

def extract_text_with_links(pdf_path):
    """Extract text and hyperlinks from a PDF."""
    doc = fitz.open(pdf_path)
    extracted_data = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        links = []

        # Extract clickable links
        for link in page.get_links():
            if "uri" in link:
                links.append(link["uri"])
                page_text += f"\n[🔗 Link: {link['uri']}]"

        # Extract plain text URLs using regex
        url_pattern = r"https?://\S+"  
        text_links = re.findall(url_pattern, page_text)
        links.extend(text_links)

        # Remove duplicates
        links = list(set(links))

        extracted_data.append({
            "page": page_num + 1,
            "text": page_text,
            "links": links
        })

    return extracted_data

def get_google_oauth_url():
    """Generate Google OAuth URL for authentication."""
    try:
        # Validate environment variables
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            raise ValueError("Google OAuth credentials not found in environment variables")
        
        client_config = {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": ["http://localhost:8501"]
            }
        }
        
        flow = Flow.from_client_config(client_config, SCOPES)
        flow.redirect_uri = "http://localhost:8501"
        
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        # Store flow in session for later use
        st.session_state.oauth_flow = flow
        
        return auth_url
        
    except Exception as e:
        st.error(f"Error generating OAuth URL: {str(e)}")
        return None

def authenticate_with_google(callback_url):
    """Authenticate user with Google using callback URL."""
    try:
        # Re-create flow if not in session state
        if 'oauth_flow' not in st.session_state:
            # Recreate the flow
            if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
                st.error("Google OAuth credentials not found in environment variables")
                return None, None
            
            client_config = {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": ["http://localhost:8501"]
                }
            }
            
            flow = Flow.from_client_config(client_config, SCOPES)
            flow.redirect_uri = "http://localhost:8501"
            st.session_state.oauth_flow = flow
        else:
            flow = st.session_state.oauth_flow
        
        # Validate callback URL format
        if not callback_url.startswith('http://localhost:8501'):
            st.error("Invalid callback URL received.")
            return None, None
            
        if 'code=' not in callback_url:
            st.error("No authorization code found. Please try signing in again.")
            return None, None
        
        # Exchange authorization code for credentials
        # Suppress scope mismatch warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flow.fetch_token(authorization_response=callback_url)
        
        credentials = flow.credentials
        
        # Get user info
        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        
        return credentials, user_info
        
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return None, None

def scan_entire_drive_for_pdfs(credentials):
    """Scan user's entire Google Drive for PDF files."""
    try:
        service = build('drive', 'v3', credentials=credentials)
        
        st.session_state.processing_status = "📁 Scanning your Google Drive for PDF files..."
        
        # Query to find all PDF files in the user's drive
        query = "mimeType='application/pdf' and trashed=false"
        results = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, parents, size, modifiedTime, webViewLink)",
            pageSize=1000  # Maximum allowed
        ).execute()
        
        files = results.get('files', [])
        
        # Handle pagination if there are more files
        while 'nextPageToken' in results:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, parents, size, modifiedTime, webViewLink)",
                pageSize=1000,
                pageToken=results['nextPageToken']
            ).execute()
            files.extend(results.get('files', []))
        
        st.session_state.processing_status = f"📊 Found {len(files)} PDF files in your Drive"
        return files, service
    
    except Exception as e:
        st.error(f"Failed to scan Google Drive: {str(e)}")
        return [], None

def download_pdf_from_drive(service, file_id, file_name, download_dir):
    """Download a single PDF file from Google Drive."""
    try:
        request = service.files().get_media(fileId=file_id)
        file_path = os.path.join(download_dir, f"{file_id}_{file_name}")
        
        with open(file_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
        
        return file_path
    except Exception as e:
        print(f"Failed to download {file_name}: {str(e)}")
        return None

def get_user_vectorstore_path(user_email):
    """Get user-specific vectorstore path."""
    user_hash = hashlib.md5(user_email.encode()).hexdigest()
    return f"vectorstores/user_{user_hash}"

def get_user_metadata_path(user_email):
    """Get user-specific metadata path for tracking processed files."""
    user_hash = hashlib.md5(user_email.encode()).hexdigest()
    return f"vectorstores/user_{user_hash}_metadata.json"

def save_user_vectorstore(vectorstore, user_email):
    """Save user's vectorstore to disk."""
    path = get_user_vectorstore_path(user_email)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vectorstore.save_local(path)

def save_processed_files_metadata(file_metadata, user_email):
    """Save metadata about processed files."""
    metadata_path = get_user_metadata_path(user_email)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(file_metadata, f, indent=2)

def load_processed_files_metadata(user_email):
    """Load metadata about previously processed files."""
    metadata_path = get_user_metadata_path(user_email)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load metadata: {e}")
            return {}
    return {}

def load_user_vectorstore(user_email):
    """Load user's existing vectorstore from disk."""
    path = get_user_vectorstore_path(user_email)
    if os.path.exists(path):
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Failed to load existing vectorstore: {e}")
            return None
    return None

def process_all_user_pdfs(credentials, user_email):
    """Process all PDF files from user's Google Drive with smart incremental updates."""
    
    # Always scan the drive first to check for new files
    st.session_state.processing_status = "🔍 Scanning your Google Drive..."
    pdf_files, service = scan_entire_drive_for_pdfs(credentials)
    
    if not pdf_files:
        st.warning("No PDF files found in your Google Drive.")
        return None
    
    # Load existing vectorstore and processed files metadata
    existing_vectorstore = load_user_vectorstore(user_email)
    processed_files_metadata = load_processed_files_metadata(user_email)
    
    # Create a map of currently processed files (file_id -> metadata)
    processed_files_map = {item['file_id']: item for item in processed_files_metadata.get('files', [])}
    
    # Find new files that haven't been processed yet
    new_files = []
    updated_files = []
    
    for file_info in pdf_files:
        file_id = file_info['id']
        current_modified_time = file_info.get('modifiedTime', '')
        
        if file_id not in processed_files_map:
            # Completely new file
            new_files.append(file_info)
        elif processed_files_map[file_id].get('modified_time') != current_modified_time:
            # File has been modified since last processing
            updated_files.append(file_info)
    
    files_to_process = new_files + updated_files
    
    # If no new or updated files, just load existing vectorstore
    if not files_to_process and existing_vectorstore:
        st.info(f"📚 Loading your existing document collection ({len(processed_files_map)} files already processed)")
        st.session_state.vectorstore = existing_vectorstore
        return existing_vectorstore
    
    # Process new/updated files
    if files_to_process:
        st.info(f"🔄 Processing {len(new_files)} new files and {len(updated_files)} updated files...")
        
        documents = []
        processed_count = 0
        failed_count = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_info in enumerate(files_to_process):
                file_id = file_info['id']
                file_name = file_info['name']
                
                try:
                    # Update progress
                    progress = (i + 1) / len(files_to_process)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{len(files_to_process)}: {file_name}")
                    
                    # Download PDF
                    pdf_path = download_pdf_from_drive(service, file_id, file_name, temp_dir)
                    
                    if pdf_path and os.path.exists(pdf_path):
                        # Extract text with links
                        extracted_pages = extract_text_with_links(pdf_path)
                        
                        # Process each page
                        from langchain_core.documents import Document
                        for page in extracted_pages:
                            combined_text = page["text"] + "\n\n🔗 Links:\n" + "\n".join(page["links"])
                            metadata = {
                                "source": file_name,
                                "page": page["page"],
                                "file_id": file_id,
                                "drive_link": file_info.get('webViewLink', ''),
                                "modified_time": file_info.get('modifiedTime', ''),
                                "user_email": user_email
                            }
                            documents.append(Document(page_content=combined_text, metadata=metadata))
                        
                        processed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    failed_count += 1
                    print(f"Failed to process {file_name}: {str(e)}")
                    continue
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        if not documents:
            if existing_vectorstore:
                st.info("📚 No new documents to process. Loading existing collection...")
                st.session_state.vectorstore = existing_vectorstore
                return existing_vectorstore
            else:
                st.error("No documents were processed successfully.")
                return None
        
        st.success(f"✅ Successfully processed {processed_count} files ({failed_count} failed)")
        st.info("🧠 Creating embeddings...")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )
        splits = text_splitter.split_documents(documents)
        
        st.info(f"📊 Created {len(splits)} text chunks from new/updated documents")
        
        # Create embeddings for new documents
        with st.spinner("Creating embeddings..."):
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
            
            if existing_vectorstore and len(updated_files) > 0:
                # If we have updated files, we need to remove old versions first
                # For FAISS, it's easier to rebuild the vectorstore completely
                # Include all files in the new vectorstore
                st.info("🔄 Rebuilding vectorstore with updated files...")
                
                # Get all current files (not just new ones) 
                all_documents = []
                with tempfile.TemporaryDirectory() as temp_dir:
                    for file_info in pdf_files:
                        file_id = file_info['id']
                        file_name = file_info['name']
                        
                        try:
                            pdf_path = download_pdf_from_drive(service, file_id, file_name, temp_dir)
                            if pdf_path and os.path.exists(pdf_path):
                                extracted_pages = extract_text_with_links(pdf_path)
                                from langchain_core.documents import Document
                                for page in extracted_pages:
                                    combined_text = page["text"] + "\n\n🔗 Links:\n" + "\n".join(page["links"])
                                    metadata = {
                                        "source": file_name,
                                        "page": page["page"],
                                        "file_id": file_id,
                                        "drive_link": file_info.get('webViewLink', ''),
                                        "modified_time": file_info.get('modifiedTime', ''),
                                        "user_email": user_email
                                    }
                                    all_documents.append(Document(page_content=combined_text, metadata=metadata))
                        except Exception as e:
                            print(f"Failed to reprocess {file_name}: {str(e)}")
                            continue
                
                all_splits = text_splitter.split_documents(all_documents)
                vectorstore = FAISS.from_documents(all_splits, embeddings)
                
            elif existing_vectorstore:
                # Only new files, add to existing vectorstore
                new_vectorstore = FAISS.from_documents(splits, embeddings)
                existing_vectorstore.merge_from(new_vectorstore)
                vectorstore = existing_vectorstore
            else:
                # No existing vectorstore, create new one
                vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Save vectorstore and update metadata
        save_user_vectorstore(vectorstore, user_email)
        
        # Update processed files metadata
        updated_metadata = {
            'files': [
                {
                    'file_id': file_info['id'],
                    'file_name': file_info['name'],
                    'modified_time': file_info.get('modifiedTime', ''),
                    'processed_at': datetime.now().isoformat()
                }
                for file_info in pdf_files
            ],
            'last_scan': datetime.now().isoformat(),
            'total_files': len(pdf_files)
        }
        save_processed_files_metadata(updated_metadata, user_email)
        
        st.session_state.vectorstore = vectorstore
        st.success("🎉 Your document collection is up to date!")
        return vectorstore
    
    # This shouldn't happen, but just in case
    if existing_vectorstore:
        st.session_state.vectorstore = existing_vectorstore
        return existing_vectorstore
    
    return None

def setup_qa_chain(vectorstore):
    """Set up the QA chain with FAISS retriever."""
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

    # Set up retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Define prompt for history-aware retrieval
    prompt_for_history = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Generate a search query based on the conversation history and question.")
    ])

    # Define answer prompt (force LLM to include links)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful technical assistant. Provide clear, accurate answers based on the documentation.
        - If the context contains URLs, **always include them**.
        - Format URLs as proper clickable links.
        - If referring to a specific section with a link, mention both the section name and the link.

        Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # Create chains
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_for_history)

    return create_retrieval_chain(history_aware_retriever, document_chain)

def main():
    st.set_page_config(
        page_title="Personal AI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Personal AI Assistant with Google Drive")

    # Handle OAuth callback automatically
    query_params = st.query_params
    if 'code' in query_params and not st.session_state.authenticated:
        st.info("🔄 Processing authentication...")
        
        # Construct the full callback URL
        callback_url = f"http://localhost:8501?code={query_params['code']}"
        if 'state' in query_params:
            callback_url += f"&state={query_params['state']}"
        if 'scope' in query_params:
            callback_url += f"&scope={query_params['scope']}"
            
        # Process authentication
        with st.spinner("Completing sign in..."):
            credentials, user_info = authenticate_with_google(callback_url)
            if credentials and user_info:
                st.session_state.credentials = credentials
                st.session_state.user_info = user_info
                st.session_state.authenticated = True
                st.success(f"Welcome, {user_info.get('name', 'User')}!")
                # Clear query params and redirect to clean URL
                st.query_params.clear()
                st.rerun()
            else:
                st.error("Authentication failed. Please try again.")
                st.query_params.clear()

    # Debug info (remove after fixing)
    if st.checkbox("Show Debug Info"):
        st.write("Session State:")
        st.write(f"authenticated: {st.session_state.authenticated}")
        st.write(f"user_info: {st.session_state.user_info}")
        st.write(f"credentials: {st.session_state.credentials is not None}")
        st.write(f"Query params: {dict(query_params)}")
        
        # Test button to manually force authentication state
        if st.button("Force Authentication (Debug Only)"):
            st.session_state.authenticated = True
            st.session_state.user_info = {"name": "Test User", "email": "test@example.com"}
            st.rerun()
            
        # Clear OAuth flow button
        if st.button("Clear OAuth Flow (Debug Only)"):
            if 'oauth_flow' in st.session_state:
                del st.session_state.oauth_flow
            st.success("OAuth flow cleared")

    # Check authentication
    if not st.session_state.authenticated:
        # Login page
        st.markdown("### 🔐 Sign in with Google to access your documents")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            **This AI assistant will:**
            - 🔍 Scan your entire Google Drive for PDF documents
            - 🧠 Create embeddings from all your documents
            - 💬 Let you chat with your personal document collection
            - 🔒 Keep your data secure and private
            
            **To get started:**
            1. Click the button below to sign in with Google
            2. Grant permissions to read your Google Drive
            3. Wait while we process your documents
            4. Start chatting with your AI assistant!
            """)
            
            if st.button("🚀 Sign in with Google", type="primary", use_container_width=True):
                try:
                    auth_url = get_google_oauth_url()
                    if auth_url:
                        st.markdown(f"""
                        **[🔗 Click here to authenticate with Google]({auth_url})**
                        
                        *You'll be redirected back automatically after signing in.*
                        """)
                        st.info("💡 After clicking the link above, you'll be taken to Google's sign-in page. Once you authorize the app, you'll be automatically redirected back here!")
                except Exception as e:
                    st.error(f"Error generating auth URL: {str(e)}")
        return

    # User is authenticated - show main interface
    with st.sidebar:
        st.header("👤 User Profile")
        if st.session_state.user_info:
            st.write(f"**Name:** {st.session_state.user_info.get('name', 'Unknown')}")
            st.write(f"**Email:** {st.session_state.user_info.get('email', 'Unknown')}")
            
            if st.button("🚪 Sign Out"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        st.divider()
        
        # Document processing section
        st.header("📚 Document Processing")
        
        if not st.session_state.processed_pdfs:
            st.info("Your documents haven't been processed yet.")
            if st.button("🔄 Process My Google Drive", type="primary"):
                user_email = st.session_state.user_info.get('email')
                with st.spinner("Processing your Google Drive..."):
                    try:
                        vectorstore = process_all_user_pdfs(st.session_state.credentials, user_email)
                        if vectorstore:
                            st.session_state.qa_chain = setup_qa_chain(vectorstore)
                            st.session_state.processed_pdfs = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
        else:
            st.success("✅ Documents processed and ready!")
            
            if st.button("🔄 Refresh Documents"):
                user_email = st.session_state.user_info.get('email')
                # Clear existing vectorstore and metadata
                vectorstore_path = get_user_vectorstore_path(user_email)
                metadata_path = get_user_metadata_path(user_email)
                
                if os.path.exists(vectorstore_path):
                    import shutil
                    shutil.rmtree(vectorstore_path)
                
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                st.session_state.processed_pdfs = False
                st.session_state.vectorstore = None
                st.rerun()
            
            if st.button("💬 Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

    # Main chat interface
    if not st.session_state.processed_pdfs:
        st.info("🔄 Please process your Google Drive documents to start chatting.")
        st.markdown("""
        ### What happens when you process your Drive:
        
        1. **🔍 Document Discovery**: We'll scan your entire Google Drive for PDF files
        2. **📄 Text Extraction**: Extract text content from all your PDFs
        3. **🧠 AI Processing**: Create smart embeddings that understand your document content
        4. **💾 Personal Database**: Build your personal knowledge base
        5. **💬 Chat Ready**: Start asking questions about any of your documents!
        
        **Your data stays private** - we only read your documents to build your personal AI assistant.
        """)
        return

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Retrieve documents from FAISS
        retriever = st.session_state.vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(prompt)

        # Debug retrieved chunks
        print("\n=== Retrieved Chunks ===")
        for i, doc in enumerate(retrieved_docs[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:\n", doc.page_content[:2000])  # First 2000 chars

        # Generate response from LLM
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                formatted_history = [(msg["role"], msg["content"]) for msg in st.session_state.chat_history[:-1]]
                response = st.session_state.qa_chain.invoke({
                    "chat_history": formatted_history,
                    "input": prompt
                })

                print("\n=== LLM Final Answer ===\n", response["answer"])
                st.write(response["answer"])

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()
