import streamlit as st
import os
import pickle
import fitz  # PyMuPDF for PDF processing
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import re
from opensearchpy import OpenSearch
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
import hashlib
import json
from datetime import datetime
import uuid
import base64
from PIL import Image
import io
from image_processor import init_gemini, process_pdf_images

# LlamaIndex imports for sheets processing
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings, VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_parse import LlamaParse
import traceback

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_OAUTH_CLIENT_SECRET')

# Allow insecure transport for local development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# OpenSearch configuration
OPENSEARCH_URL = os.getenv('OPENSEARCH_URL', 'http://localhost:9200')
OPENSEARCH_USERNAME = os.getenv('OPENSEARCH_USERNAME', '')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', '')
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'document_embeddings')

# Google OAuth configuration
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
if 'processed_sheets' not in st.session_state:
    st.session_state.processed_sheets = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'sheets_query_engine' not in st.session_state:
    st.session_state.sheets_query_engine = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""
if 'start_chatting' not in st.session_state:
    st.session_state.start_chatting = False

def create_opensearch_client():
    """Create OpenSearch client connection."""
    try:
        # For local OpenSearch without authentication
        if not OPENSEARCH_USERNAME and not OPENSEARCH_PASSWORD:
            client = OpenSearch(
                hosts=[OPENSEARCH_URL],
                use_ssl=False,
                verify_certs=False,
                ssl_show_warn=False
            )
        else:
            # For OpenSearch with authentication
            client = OpenSearch(
                hosts=[OPENSEARCH_URL],
                http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
                use_ssl=OPENSEARCH_URL.startswith('https'),
                verify_certs=False,
                ssl_show_warn=False
            )
        return client
    except Exception as e:
        st.error(f"Failed to connect to OpenSearch: {str(e)}")
        return None

def create_opensearch_index(client):
    """Create OpenSearch index with proper mapping for vector search."""
    try:
        # Delete existing index if it exists
        if client.indices.exists(index=OPENSEARCH_INDEX):
            print(f"🗑️ Deleting existing index: {OPENSEARCH_INDEX}")
            client.indices.delete(index=OPENSEARCH_INDEX)
        
        # Define the index mapping for vector search
        index_mapping = {
            "mappings": {
                "properties": {
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 768,  # Gemini embedding dimension
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene"
                        }
                    },
                    "text": {
                        "type": "text"
                    },
                    "metadata": {
                        "type": "object"
                    }
                }
            },
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            }
        }
        
        # Create the index
        client.indices.create(
            index=OPENSEARCH_INDEX,
            body=index_mapping
        )
        
        print(f"✅ Created OpenSearch index: {OPENSEARCH_INDEX}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating OpenSearch index: {str(e)}")
        return False

def check_existing_documents():
    """Check if there are existing documents in OpenSearch."""
    try:
        client = create_opensearch_client()
        if not client:
            return False
        
        # Check if index exists
        if not client.indices.exists(index=OPENSEARCH_INDEX):
            return False
        
        # Count documents in the index
        count_response = client.count(index=OPENSEARCH_INDEX)
        document_count = count_response.get('count', 0)
        
        return document_count > 0
        
    except Exception as e:
        print(f"Error checking existing documents: {str(e)}")
        return False

def initialize_qa_chain_from_existing():
    """Initialize QA chain from existing OpenSearch documents."""
    try:
        print("🔧 Starting QA chain initialization...")
        
        # Create embeddings
        print("📝 Creating embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Create vector store connection to existing index
        print("🔗 Connecting to OpenSearch vector store...")
        vectorstore = OpenSearchVectorSearch(
            embedding_function=embeddings,
            opensearch_url=OPENSEARCH_URL,
            index_name=OPENSEARCH_INDEX,
            vector_field="vector_field",
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD) if OPENSEARCH_USERNAME else None,
            use_ssl=OPENSEARCH_URL.startswith('https'),
            verify_certs=False,
            engine="lucene"
        )
        
        # Set up QA chain
        print("⚙️ Setting up QA chain...")
        qa_chain = setup_qa_chain(vectorstore)
        
        # Update session state
        print("💾 Updating session state...")
        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = qa_chain
        st.session_state.start_chatting = True
        
        print("✅ QA chain initialization completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing QA chain: {str(e)}")
        st.error(f"Error initializing QA chain: {str(e)}")
        return False

def extract_text_with_links(pdf_path):
    """Extract text and hyperlinks from a PDF."""
    try:
        doc = fitz.open(pdf_path)
        extracted_data = []
        
        print(f"📖 PDF has {len(doc)} pages")

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
            
            # Debug: Show text length for each page
            text_length = len(page_text.strip())
            print(f"📄 Page {page_num + 1}: {text_length} characters, {len(links)} links")

            extracted_data.append({
                "page": page_num + 1,
                "text": page_text,
                "links": links
            })
        
        doc.close()
        return extracted_data
        
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

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
        
        # Configure flow with insecure transport for local development
        flow = Flow.from_client_config(
            client_config, 
            SCOPES,
            redirect_uri="http://localhost:8501"
        )
        
        # Set additional flow properties
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
            
            # Configure flow with insecure transport for local development
            flow = Flow.from_client_config(
                client_config, 
                SCOPES,
                redirect_uri="http://localhost:8501"
            )
            st.session_state.oauth_flow = flow
        else:
            flow = st.session_state.oauth_flow
        
        # Validate callback URL format
        if not isinstance(callback_url, str) or not callback_url.startswith('http://localhost:8501'):
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
    """Scan entire Google Drive for PDF files."""
    try:
        service = build('drive', 'v3', credentials=credentials)
        
        # Query to find all PDF files
        query = "mimeType='application/pdf' and trashed=false"
        
        results = []
        page_token = None
        
        while True:
            response = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, modifiedTime, size)',
                pageToken=page_token
            ).execute()
            
            results.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            
            if page_token is None:
                break
        
        return results
        
    except Exception as e:
        st.error(f"Error scanning Drive for PDFs: {str(e)}")
        return []

def scan_entire_drive_for_docs(credentials):
    """Scan entire Google Drive for Google Docs files."""
    try:
        service = build('drive', 'v3', credentials=credentials)
        
        # Query to find all Google Docs files
        query = "mimeType='application/vnd.google-apps.document' and trashed=false"
        
        results = []
        page_token = None
        
        while True:
            response = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, modifiedTime, size)',
                pageToken=page_token
            ).execute()
            
            results.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            
            if page_token is None:
                break
        
        return results
        
    except Exception as e:
        st.error(f"Error scanning Drive for Google Docs: {str(e)}")
        return []

def scan_entire_drive_for_sheets(credentials):
    """Scan entire Google Drive for Excel and Google Sheets files."""
    try:
        service = build('drive', 'v3', credentials=credentials)
        
        # Query to find Excel files and Google Sheets
        excel_query = "mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' and trashed=false"
        google_sheets_query = "mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
        
        results = []
        
        # Get Excel files
        page_token = None
        while True:
            response = service.files().list(
                q=excel_query,
                spaces='drive',
                fields='nextPageToken, files(id, name, modifiedTime, size)',
                pageToken=page_token
            ).execute()
            
            results.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            
            if page_token is None:
                break
        
        # Get Google Sheets
        page_token = None
        while True:
            response = service.files().list(
                q=google_sheets_query,
                spaces='drive',
                fields='nextPageToken, files(id, name, modifiedTime, size)',
                pageToken=page_token
            ).execute()
            
            results.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            
            if page_token is None:
                break
        
        return results
        
    except Exception as e:
        st.error(f"Error scanning Drive for sheets: {str(e)}")
        return []

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

def download_google_doc_content(service, file_id, file_name):
    """Download Google Doc content as text."""
    try:
        # Export Google Doc as plain text
        request = service.files().export_media(
            fileId=file_id,
            mimeType='text/plain'
        )
        
        # Download the content
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        
        while done is False:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download {int(status.progress() * 100)}%")
        
        content = fh.getvalue().decode('utf-8')
        
        # Create a document object similar to PDF processing
        from langchain_core.documents import Document
        
        doc = Document(
            page_content=content,
            metadata={
                'source': file_name,
                'type': 'text',
                'file_id': file_id,
                'file_type': 'google_doc',
                'processed_time': datetime.now().isoformat()
            }
        )
        
        return doc
        
    except Exception as e:
        st.error(f"Error downloading Google Doc {file_name}: {str(e)}")
        return None

def download_sheet_from_drive(service, file_id, file_name, download_dir):
    """Download a sheet file from Google Drive."""
    try:
        # For Excel files, download directly
        if file_name.endswith(('.xlsx', '.xls')):
            request = service.files().get_media(fileId=file_id)
        else:
            # For Google Sheets, export as Excel
            request = service.files().export_media(
                fileId=file_id,
                mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        
        file_path = os.path.join(download_dir, f"{file_id}_{file_name}")
        if not file_path.endswith('.xlsx'):
            file_path += '.xlsx'
        
        with open(file_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
        
        return file_path
    except Exception as e:
        print(f"Failed to download {file_name}: {str(e)}")
        return None

def process_pdf_with_images(pdf_path, file_name, file_id):
    """Process a PDF file to extract both text and images."""
    try:
        # Initialize Gemini Vision model
        model = init_gemini(GOOGLE_API_KEY)
        
        documents = []
        
        # Extract text with links
        print(f"🔍 Extracting text from PDF: {file_name}")
        extracted_pages = extract_text_with_links(pdf_path)
        print(f"📄 Extracted {len(extracted_pages)} pages from {file_name}")
        
        # Process text documents
        from langchain_core.documents import Document
        text_doc_count = 0
        for page in extracted_pages:
            page_text = page["text"].strip()
            if page_text:  # Only create document if there's actual text
                combined_text = page_text + "\n\n🔗 Links:\n" + "\n".join(page["links"])
                metadata = {
                    "source": file_name,
                    "page": page["page"],
                    "file_id": file_id,
                    "type": "text"
                }
                documents.append(Document(page_content=combined_text, metadata=metadata))
                text_doc_count += 1
                print(f"✅ Created text document for page {page['page']} ({len(page_text)} characters)")
            else:
                print(f"⚠️ No text found on page {page['page']}")
        
        print(f"📝 Created {text_doc_count} text documents from {file_name}")
        
        # Process images using Gemini Vision
        print(f"🖼️ Starting image processing for {file_name}")
        image_results = process_pdf_images(pdf_path, model)
        print(f"🖼️ Image processing completed for {file_name}: {len(image_results)} results")
        
        image_doc_count = 0
        for img_result in image_results:
            metadata = {
                "source": file_name,
                "page": img_result["page"],
                "file_id": file_id,
                "image_id": img_result["image_id"],
                "type": "image"
            }
            documents.append(Document(page_content=img_result["description"], metadata=metadata))
            image_doc_count += 1
            print(f"✅ Added image document: {img_result['image_id']}")
        
        print(f"📊 Total documents created for {file_name}: {len(documents)} ({text_doc_count} text, {image_doc_count} images)")
        
        return documents
        
    except Exception as e:
        print(f"❌ Error processing PDF {file_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_user_vectorstore_path(user_email):
    """Get user-specific vectorstore path."""
    user_hash = hashlib.md5(user_email.encode()).hexdigest()
    return f"vectorstores/user_{user_hash}"

def get_user_metadata_path(user_email):
    """Get user-specific metadata path for tracking processed files."""
    user_hash = hashlib.md5(user_email.encode()).hexdigest()
    return f"vectorstores/user_{user_hash}_metadata.json"

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

def process_all_user_documents(credentials, user_email):
    """Process all PDF, Google Doc, and Sheet files for a user."""
    try:
        # Create OpenSearch client
        client = create_opensearch_client()
        if not client:
            raise Exception("Failed to create OpenSearch client")

        # Check if index exists, if not create it
        if not client.indices.exists(index=OPENSEARCH_INDEX):
            create_opensearch_index(client)

        # Load previously processed files metadata
        processed_files = load_processed_files_metadata(user_email)
        
        # Scan for PDFs, Google Docs, and Sheets
        st.session_state.processing_status = "📁 Scanning your Google Drive for PDF, Google Doc, and Sheet files..."
        
        pdf_files = scan_entire_drive_for_pdfs(credentials)
        doc_files = scan_entire_drive_for_docs(credentials)
        sheet_files = scan_entire_drive_for_sheets(credentials)
        
        all_files = pdf_files + doc_files + sheet_files
        st.session_state.processing_status = f"📊 Found {len(pdf_files)} PDF files, {len(doc_files)} Google Doc files, and {len(sheet_files)} Sheet files in your Drive"
        
        if not all_files:
            st.warning("No PDF, Google Doc, or Sheet files found in your Google Drive.")
            
            # Check if there are existing documents in OpenSearch
            existing_doc_count = get_document_count()
            if existing_doc_count > 0:
                st.info(f"📚 Found {existing_doc_count} existing documents in your knowledge base.")
                st.success("✅ You can start asking questions about your existing documents!")
                
                # Initialize QA chain from existing documents
                if initialize_qa_chain_from_existing():
                    st.session_state.processed_pdfs = True
                    st.session_state.processing_status = "✅ Ready to chat with existing documents!"
                else:
                    st.error("❌ Failed to initialize chat interface with existing documents.")
            else:
                st.session_state.processing_status = "📚 No documents found. Please add documents to your Google Drive."
            
            return

        # Build Drive service
        service = build('drive', 'v3', credentials=credentials)
        
        # Process files
        documents = []
        all_sheet_documents = []  # Store documents instead of file paths
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        print(f"🔍 Processing {len(pdf_files)} PDF files, {len(doc_files)} Google Doc files, and {len(sheet_files)} Sheet files")
        
        # Process PDFs
        for pdf_file in pdf_files:
            file_id = pdf_file['id']
            file_name = pdf_file['name']
            
            # Check if already processed
            file_hash = hashlib.md5(f"{file_id}_{pdf_file.get('modifiedTime', '')}".encode()).hexdigest()
            if file_hash in processed_files:
                print(f"⏭️ Skipping already processed PDF: {file_name}")
                skipped_count += 1
                continue
            
            try:
                st.session_state.processing_status = f"📄 Processing PDF: {file_name}"
                
                # Download PDF
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = download_pdf_from_drive(service, file_id, file_name, temp_dir)
                    if not pdf_path:
                        failed_count += 1
                        continue
                    
                    # Extract text and images
                    pdf_docs = process_pdf_with_images(pdf_path, file_name, file_id)
                    if pdf_docs:
                        documents.extend(pdf_docs)
                        processed_count += 1
                        
                        # Save metadata
                        processed_files[file_hash] = {
                            'file_id': file_id,
                            'file_name': file_name,
                            'file_type': 'pdf',
                            'processed_time': datetime.now().isoformat(),
                            'document_count': len(pdf_docs)
                        }
                    else:
                        failed_count += 1
                        
            except Exception as e:
                print(f"Error processing PDF {file_name}: {str(e)}")
                failed_count += 1
        
        # Process Google Docs
        for doc_file in doc_files:
            file_id = doc_file['id']
            file_name = doc_file['name']
            
            # Check if already processed
            file_hash = hashlib.md5(f"{file_id}_{doc_file.get('modifiedTime', '')}".encode()).hexdigest()
            if file_hash in processed_files:
                print(f"⏭️ Skipping already processed Google Doc: {file_name}")
                skipped_count += 1
                continue
            
            try:
                st.session_state.processing_status = f"📝 Processing Google Doc: {file_name}"
                
                # Download Google Doc content
                doc = download_google_doc_content(service, file_id, file_name)
                if doc:
                    documents.append(doc)
                    processed_count += 1
                    
                    # Save metadata
                    processed_files[file_hash] = {
                        'file_id': file_id,
                        'file_name': file_name,
                        'file_type': 'google_doc',
                        'processed_time': datetime.now().isoformat(),
                        'document_count': 1
                    }
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"Error processing Google Doc {file_name}: {str(e)}")
                failed_count += 1
        
        # Process Sheets
        for sheet_file in sheet_files:
            file_id = sheet_file['id']
            file_name = sheet_file['name']
            
            # Check if already processed
            file_hash = hashlib.md5(f"{file_id}_{sheet_file.get('modifiedTime', '')}".encode()).hexdigest()
            if file_hash in processed_files:
                print(f"⏭️ Skipping already processed Sheet: {file_name}")
                skipped_count += 1
                continue
            
            try:
                st.session_state.processing_status = f"📊 Processing Sheet: {file_name}"
                
                # Download and process sheet file immediately
                with tempfile.TemporaryDirectory() as temp_dir:
                    sheet_path = download_sheet_from_drive(service, file_id, file_name, temp_dir)
                    if sheet_path:
                        # Process with LlamaParse immediately while file exists
                        try:
                            parser = initialize_llamaindex_parser()
                            if parser:
                                sheet_documents = parser.load_data(sheet_path)
                                if sheet_documents:
                                    all_sheet_documents.extend(sheet_documents)
                                    processed_count += 1
                                    print(f"✅ Successfully processed {file_name}: {len(sheet_documents)} documents")
                                    
                                    # Save metadata
                                    processed_files[file_hash] = {
                                        'file_id': file_id,
                                        'file_name': file_name,
                                        'file_type': 'sheet',
                                        'processed_time': datetime.now().isoformat(),
                                        'document_count': len(sheet_documents)
                                    }
                                else:
                                    print(f"⚠️ No documents extracted from {file_name}")
                                    failed_count += 1
                            else:
                                print(f"❌ Failed to initialize parser for {file_name}")
                                failed_count += 1
                        except Exception as e:
                            print(f"❌ Error processing {file_name} with LlamaParse: {str(e)}")
                            failed_count += 1
                    else:
                        print(f"❌ Failed to download {file_name}")
                        failed_count += 1
                        
            except Exception as e:
                print(f"Error processing Sheet {file_name}: {str(e)}")
                failed_count += 1
        
        # Process documents with LangChain (PDFs and Google Docs)
        if documents:
            st.success(f"✅ Successfully processed {processed_count} files ({failed_count} failed)")
            st.info("🧠 Creating embeddings for documents...")
            
            # Create embeddings using Gemini (needed for semantic chunking)
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            
            # Separate text and image documents
            text_docs = [doc for doc in documents if doc.metadata.get('type') == 'text']
            image_docs = [doc for doc in documents if doc.metadata.get('type') == 'image']
            
            print(f"Processing {len(text_docs)} text documents and {len(image_docs)} image documents")
            
            # Check if we have any documents to process
            if not text_docs and not image_docs:
                print("⚠️ No text or image documents to process")
                st.warning("No text or image content found in the processed files.")
            else:
                # Advanced semantic chunking function using embeddings and similarity
                def semantic_chunk_text(text, embeddings, chunk_size=1500, overlap=300, 
                                       initial_threshold=0.6, appending_threshold=0.8):
                    """Split text semantically using embeddings to find natural breakpoints"""
                    if len(text) <= chunk_size:
                        return [text]
                    
                    # Split into sentences first using multiple approaches
                    import re
                    sentences = []
                    
                    # Try multiple sentence splitting approaches
                    # 1. Split by periods followed by space and capital letter
                    period_splits = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
                    
                    # 2. Split by other sentence endings
                    sentence_endings = re.split(r'(?<=[.!?])\s+', text)
                    
                    # Use the approach that gives more reasonable sentence lengths
                    if len(period_splits) > len(sentence_endings) and all(len(s) > 10 for s in period_splits):
                        sentences = period_splits
                    else:
                        sentences = sentence_endings
                    
                    # Clean up sentences
                    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
                    
                    if len(sentences) <= 1:
                        # Fallback to recursive splitting if no sentence breaks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=overlap,
                            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
                            keep_separator=True
                        )
                        return text_splitter.split_text(text)
                    
                    # Generate embeddings for all sentences
                    print(f"Generating embeddings for {len(sentences)} sentences...")
                    sentence_embeddings = embeddings.embed_documents(sentences)
                    
                    # Calculate cosine similarity between consecutive sentences
                    import numpy as np
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    chunks = []
                    current_chunk = []
                    current_length = 0
                    
                    for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
                        sentence_length = len(sentence)
                        
                        # Check if adding this sentence would exceed chunk size
                        if current_length + sentence_length > chunk_size and current_chunk:
                            # Finalize current chunk
                            chunks.append(' '.join(current_chunk))
                            
                            # Start new chunk with overlap (last few sentences)
                            overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                            current_chunk = overlap_sentences.copy()
                            current_length = sum(len(s) for s in current_chunk)
                        
                        # If this is the first sentence or we have a current chunk, check similarity
                        if current_chunk:
                            # Calculate similarity with the last sentence in current chunk
                            last_sentence_idx = i - 1
                            if last_sentence_idx >= 0:
                                last_embedding = sentence_embeddings[last_sentence_idx]
                                similarity = cosine_similarity([last_embedding], [embedding])[0][0]
                                
                                # If similarity is low, start a new chunk (semantic breakpoint)
                                if similarity < initial_threshold and current_length > chunk_size * 0.3:
                                    chunks.append(' '.join(current_chunk))
                                    current_chunk = [sentence]
                                    current_length = sentence_length
                                    continue
                        
                        # Add sentence to current chunk
                        current_chunk.append(sentence)
                        current_length += sentence_length
                    
                    # Add the final chunk
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    
                    # Post-process: merge very similar adjacent chunks
                    if len(chunks) > 1:
                        final_chunks = []
                        i = 0
                        while i < len(chunks):
                            if i < len(chunks) - 1:
                                # Check if we can merge with next chunk
                                chunk1_embedding = embeddings.embed_documents([chunks[i]])[0]
                                chunk2_embedding = embeddings.embed_documents([chunks[i + 1]])[0]
                                similarity = cosine_similarity([chunk1_embedding], [chunk2_embedding])[0][0]
                                
                                if similarity > appending_threshold and len(chunks[i]) + len(chunks[i + 1]) <= chunk_size * 1.2:
                                    # Merge chunks
                                    merged_chunk = chunks[i] + " " + chunks[i + 1]
                                    final_chunks.append(merged_chunk)
                                    i += 2  # Skip next chunk
                                else:
                                    final_chunks.append(chunks[i])
                                    i += 1
                            else:
                                final_chunks.append(chunks[i])
                                i += 1
                        
                        chunks = final_chunks
                    
                    return chunks
                
                # Apply semantic chunking to text documents
                text_splits = []
                for doc in text_docs:
                    print(f"Semantically chunking document: {doc.metadata.get('source', 'unknown')}")
                    chunks = semantic_chunk_text(doc.page_content, embeddings)
                    for i, chunk in enumerate(chunks):
                        from langchain_core.documents import Document
                        chunk_doc = Document(
                            page_content=chunk,
                            metadata=doc.metadata.copy()
                        )
                        chunk_doc.metadata['chunk_id'] = i
                        chunk_doc.metadata['total_chunks'] = len(chunks)
                        chunk_doc.metadata['chunking_method'] = 'semantic'
                        text_splits.append(chunk_doc)
                
                # For image descriptions, group by source and page for better organization
                image_splits = []
                if image_docs:
                    # Group images by source and page
                    image_groups = {}
                    for img_doc in image_docs:
                        source = img_doc.metadata.get('source', 'unknown')
                        page = img_doc.metadata.get('page', 0)
                        key = f"{source}_page_{page}"
                        
                        if key not in image_groups:
                            image_groups[key] = []
                        image_groups[key].append(img_doc)
                    
                    # Process each group semantically
                    for key, group_docs in image_groups.items():
                        if len(group_docs) == 1:
                            # Single image, keep as-is
                            image_splits.extend(group_docs)
                        else:
                            # Multiple images on same page, combine and apply semantic chunking
                            combined_content = "\n\n".join([doc.page_content for doc in group_docs])
                            chunks = semantic_chunk_text(combined_content, embeddings, chunk_size=1000, overlap=200)
                            
                            for i, chunk in enumerate(chunks):
                                from langchain_core.documents import Document
                                chunk_doc = Document(
                                    page_content=chunk,
                                    metadata=group_docs[0].metadata.copy()
                                )
                                chunk_doc.metadata['image_count'] = len(group_docs)
                                chunk_doc.metadata['image_ids'] = [doc.metadata.get('image_id', '') for doc in group_docs]
                                chunk_doc.metadata['chunk_id'] = i
                                chunk_doc.metadata['total_chunks'] = len(chunks)
                                chunk_doc.metadata['chunking_method'] = 'semantic'
                                image_splits.append(chunk_doc)
                
                # Combine all splits
                all_splits = text_splits + image_splits
                
                if all_splits:
                    st.info(f"📊 Created {len(text_splits)} semantic text chunks from {len(text_docs)} text documents and {len(image_splits)} semantic image chunks from {len(image_docs)} image descriptions")
                    
                    # Create vector store
                    vectorstore = OpenSearchVectorSearch(
                        embedding_function=embeddings,
                        opensearch_url=OPENSEARCH_URL,
                        index_name=OPENSEARCH_INDEX,
                        vector_field="vector_field",
                        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD) if OPENSEARCH_USERNAME else None,
                        use_ssl=OPENSEARCH_URL.startswith('https'),
                        verify_certs=False,
                        engine="lucene"
                    )

                    # Add documents to vector store
                    print(f"Adding {len(all_splits)} total documents to vector store")
                    vectorstore.add_documents(all_splits)
                    
                    # Set up QA chain
                    st.session_state.vectorstore = vectorstore
                    st.session_state.qa_chain = setup_qa_chain(vectorstore)
                    st.session_state.processed_pdfs = True
                else:
                    print("⚠️ No document chunks created")
                    st.warning("No document content could be processed for embeddings.")
        else:
            # No documents to process, but we might have sheets
            if all_sheet_documents:
                st.success(f"✅ Successfully processed {processed_count} files ({failed_count} failed)")
                st.info("📊 No documents found, but sheets were processed successfully.")
            else:
                st.warning("No documents or sheets found to process.")
        
        # Process sheets with LlamaIndex
        if all_sheet_documents:
            st.info("📊 Processing sheets with LlamaIndex...")
            
            try:
                sheets_query_engine = create_sheets_query_engine(all_sheet_documents)
                if sheets_query_engine:
                    st.session_state.sheets_query_engine = sheets_query_engine
                    st.session_state.processed_sheets = True
                    st.success("✅ Sheets query engine created successfully!")
                else:
                    st.error("❌ Failed to create sheets query engine")
            except Exception as e:
                st.error(f"❌ Error creating sheets query engine: {str(e)}")
        
        # Save updated metadata
        save_processed_files_metadata(processed_files, user_email)
        
        # Update session state
        st.session_state.processing_status = "✅ Processing complete! You can now ask questions about your documents and sheets."
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        st.session_state.processing_status = f"❌ Error: {str(e)}"

def setup_qa_chain(vectorstore):
    """Set up the QA chain with OpenSearch retriever using Gemini."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

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

    # Define answer prompt (simplified to only use available variables)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful technical assistant with access to both text documents and image descriptions from PDFs. 

**IMPORTANT GUIDELINES:**
- If the context contains URLs, **always include them**.
- Format URLs as proper clickable links.
- Never ever answer a question from your own knowledge.
- If referring to a specific section with a link, mention both the section name and the link.
- You have access to both text content and image descriptions from the documents.
- **ALWAYS check image descriptions when answering questions** - they contain valuable visual information.
- When answering questions about images, use the image descriptions and metadata available.
- If someone asks about visual content, diagrams, charts, logos, or any visual elements, prioritize image descriptions.
- Always cite your sources by mentioning the document name and type (text/image).
- If the question is about visual content but no relevant images are found, mention this.

**IMAGE-RELATED KEYWORDS TO WATCH FOR:**
- "image", "picture", "photo", "screenshot", "diagram", "chart", "graph", "logo", "icon"
- "visual", "appearance", "look like", "show", "display", "depict", "illustrate"
- "dashboard", "interface", "UI", "layout", "design", "mockup", "wireframe"
- "color", "style", "format", "size", "dimensions", "resolution"

**When responding:**
1. First check if there are relevant image descriptions
2. Use both text and image information to provide comprehensive answers
3. If images are mentioned, describe what they show based on the image descriptions
4. Always mention the source document and whether information came from text or images
5. If the question contains image-related keywords, prioritize image descriptions in your response

Context: {context}"""),
        ("user", "{input}")
    ])

    # Create chains
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_for_history)

    return create_retrieval_chain(history_aware_retriever, document_chain)

def get_document_count():
    """Get the number of documents in OpenSearch."""
    try:
        print("🔍 Checking document count in OpenSearch...")
        client = create_opensearch_client()
        if not client:
            print("❌ Failed to create OpenSearch client")
            return 0
        
        # Check if index exists
        if not client.indices.exists(index=OPENSEARCH_INDEX):
            print(f"❌ Index {OPENSEARCH_INDEX} does not exist")
            return 0
        
        # Count documents in the index
        count_response = client.count(index=OPENSEARCH_INDEX)
        document_count = count_response.get('count', 0)
        print(f"📊 Found {document_count} documents in OpenSearch")
        
        return document_count
        
    except Exception as e:
        print(f"❌ Error getting document count: {str(e)}")
        return 0

def check_metadata_mismatch(user_email):
    """Check if there's a mismatch between processed metadata and OpenSearch documents."""
    try:
        # Load processed files metadata
        processed_files = load_processed_files_metadata(user_email)
        metadata_count = len(processed_files)
        
        # Get OpenSearch document count
        opensearch_count = get_document_count()
        
        print(f"📊 Metadata shows {metadata_count} processed files")
        print(f"📊 OpenSearch shows {opensearch_count} documents")
        
        if metadata_count > 0 and opensearch_count == 0:
            print("⚠️ Mismatch detected: Metadata exists but no documents in OpenSearch")
            return True, metadata_count, opensearch_count
        else:
            print("✅ Metadata and OpenSearch are in sync")
            return False, metadata_count, opensearch_count
            
    except Exception as e:
        print(f"❌ Error checking metadata mismatch: {str(e)}")
        return False, 0, 0

def clear_metadata_and_reprocess(user_email):
    """Clear metadata and allow reprocessing of all documents."""
    try:
        print("🗑️ Clearing processed files metadata...")
        metadata_path = get_user_metadata_path(user_email)
        print(f"📁 Metadata path: {metadata_path}")
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print("✅ Metadata file deleted")
        else:
            print("⚠️ Metadata file not found")
        
        # Reset session state
        print("🔄 Resetting session state...")
        st.session_state.processed_pdfs = False
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.session_state.start_chatting = False
        
        print("✅ Session state reset")
        print("✅ Clear metadata and reprocess completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing metadata: {str(e)}")
        return False

def initialize_llamaindex_parser():
    """Initialize the LlamaParse parser for sheets processing."""
    try:
        llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not llama_key:
            raise ValueError("LLAMA_CLOUD_API_KEY not found in environment variables")
        return LlamaParse(api_key=llama_key, result_type="markdown")
    except Exception as e:
        print(f"Error initializing LlamaParse: {str(e)}")
        return None

def initialize_llamaindex_llm():
    """Initialize the Google GenAI LLM for LlamaIndex."""
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return GoogleGenAI(model="gemini-2.0-flash-exp", api_key=google_api_key, temperature=0.3)
    except Exception as e:
        print(f"Error initializing LlamaIndex LLM: {str(e)}")
        return None

def initialize_llamaindex_embeddings():
    """Initialize the Google GenAI Embeddings for LlamaIndex."""
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return GoogleGenAIEmbedding(model_name="embedding-001", api_key=google_api_key)
    except Exception as e:
        print(f"Error initializing LlamaIndex embeddings: {str(e)}")
        return None

def create_sheets_query_engine(all_documents):
    """Create and return the LlamaIndex query engine for sheets processing."""
    try:
        print("🔧 Initializing LlamaIndex components...")
        
        # Initialize components
        llm = initialize_llamaindex_llm()
        if not llm:
            raise Exception("Failed to initialize LlamaIndex LLM")
        
        embeddings = initialize_llamaindex_embeddings()
        if not embeddings:
            raise Exception("Failed to initialize LlamaIndex embeddings")
        
        Settings.llm = llm
        Settings.embed_model = embeddings
        
        print(f"📊 Processing {len(all_documents)} pre-processed sheet documents...")
        
        if not all_documents:
            raise ValueError("No documents were provided for processing")
        
        print(f"📊 Total documents to process: {len(all_documents)}")
        
        # OpenSearch configuration for sheets
        sheets_endpoint = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
        sheets_idx = "sheets_embeddings"
        text_field = "content"
        embedding_field = "embedding"
        
        # Create OpenSearch client
        client = OpenSearch(
            hosts=[sheets_endpoint],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD) if OPENSEARCH_USERNAME else None,
            use_ssl=sheets_endpoint.startswith('https'),
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # Check if sheets index exists and has correct mapping
        index_exists = client.indices.exists(index=sheets_idx)
        should_recreate_index = False
        
        if index_exists:
            try:
                current_mapping = client.indices.get_mapping(index=sheets_idx)
                properties = current_mapping[sheets_idx]['mappings']['properties']
                
                if embedding_field not in properties or properties[embedding_field]['type'] != 'knn_vector':
                    print(f"Index {sheets_idx} exists but embedding field is not correctly configured. Recreating...")
                    should_recreate_index = True
                else:
                    print(f"Index {sheets_idx} exists with correct mapping.")
            except Exception as e:
                print(f"Error checking index mapping: {e}. Recreating index...")
                should_recreate_index = True
        
        # Delete index if it needs to be recreated
        if should_recreate_index and index_exists:
            client.indices.delete(index=sheets_idx)
            print(f"Deleted existing index {sheets_idx}")
        
        # Create index if it doesn't exist or was deleted
        if not index_exists or should_recreate_index:
            index_settings = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        text_field: {"type": "text"},
                        embedding_field: {
                            "type": "knn_vector",
                            "dimension": 768,
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 16
                                }
                            }
                        }
                    }
                }
            }
            client.indices.create(index=sheets_idx, body=index_settings)
            print(f"Created new index {sheets_idx} with correct mapping")
        
        # Create vector store
        vector_client = OpensearchVectorClient(
            sheets_endpoint,
            sheets_idx,
            768,
            embedding_field=embedding_field,
            text_field=text_field
        )
        vector_store = OpensearchVectorStore(vector_client)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from documents
        index = VectorStoreIndex.from_documents(
            documents=all_documents,
            storage_context=storage_context,
            embed_model=embeddings
        )
        
        # Define custom system prompt for sheets
        custom_prompt = PromptTemplate(
            "You are a helpful AI assistant that specializes in analyzing and interpreting data from Excel sheets and spreadsheets. "
            "When responding to queries:\n"
            "1. Always provide clear, concise, and accurate answers based on the retrieved context\n"
            "2. If the information is not available in the context, clearly state that\n"
            "3. Use bullet points or numbered lists when presenting multiple pieces of information\n"
            "4. Include relevant data points, numbers, or specific details when available\n"
            "5. Maintain a professional and helpful tone\n"
            "6. If asked for analysis, provide insights and explanations, not just raw data\n"
            "7. Consider the conversation history when providing context and follow-up information\n"
            "8. When dealing with numerical data, provide context and trends\n"
            "9. If asked about specific cells, rows, or columns, reference them clearly\n\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given this information, please answer the following question: {query_str}\n"
            "Answer: "
        )
        
        # Create base query engine
        base_query_engine = index.as_query_engine(
            llm=llm, 
            response_mode="tree_summarize",
            text_qa_template=custom_prompt
        )
        
        # Create conversation memory
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        
        # Create condense question prompt for conversation context
        condense_question_prompt = PromptTemplate(
            "Given the following conversation history and a new question, rephrase the new question "
            "to be a standalone question that captures all relevant context from the conversation history.\n\n"
            "Conversation History:\n{chat_history}\n\n"
            "New Question: {question}\n\n"
            "Standalone Question: "
        )
        
        # Create chat engine with memory
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=base_query_engine,
            condense_question_prompt=condense_question_prompt,
            memory=memory,
            llm=llm,
            verbose=True
        )
        
        print("✅ Sheets query engine created successfully")
        return chat_engine
        
    except Exception as e:
        print(f"❌ Error creating sheets query engine: {str(e)}")
        print("Detailed error:")
        print(traceback.format_exc())
        return None

def combine_responses_from_dual_systems(prompt, qa_chain_response, sheets_response, llm):
    """Combine responses from both LangChain QA chain and LlamaIndex sheets query engine."""
    try:
        # Create a prompt to combine the responses
        combine_prompt = f"""
You are an AI assistant that combines information from two different sources:
1. Document/PDF/Image analysis (LangChain response)
2. Spreadsheet/Excel data analysis (LlamaIndex response)

Your task is to create a comprehensive, well-structured response that:
- Integrates information from both sources when relevant
- Eliminates redundancy while preserving important details
- Maintains clarity and organization
- Prioritizes the most relevant information for the user's question
- Clearly indicates which source provided which information when helpful

User Question: {prompt}

Document/PDF/Image Analysis Response:
{qa_chain_response if qa_chain_response else "No relevant information found in documents, PDFs, or images."}

Spreadsheet/Excel Data Analysis Response:
{sheets_response if sheets_response else "No relevant information found in spreadsheets or Excel files."}

Please provide a comprehensive answer that combines these responses intelligently:
"""
        
        # Get combined response from LLM
        response = llm.invoke(combine_prompt)
        return response.content
        
    except Exception as e:
        print(f"Error combining responses: {str(e)}")
        # Fallback: return the better of the two responses
        if qa_chain_response and sheets_response:
            return f"**Document Analysis:**\n{qa_chain_response}\n\n**Spreadsheet Analysis:**\n{sheets_response}"
        elif qa_chain_response:
            return qa_chain_response
        elif sheets_response:
            return sheets_response
        else:
            return "I couldn't find relevant information in your documents or spreadsheets."

def query_dual_systems(prompt, qa_chain, sheets_query_engine):
    """Query both LangChain QA chain and LlamaIndex sheets query engine."""
    try:
        qa_response = None
        sheets_response = None
        
        # Query LangChain QA chain (for PDFs, images, docs)
        if qa_chain:
            try:
                print("🔍 Querying LangChain QA chain...")
                qa_result = qa_chain.invoke({"input": prompt})
                qa_response = qa_result.get("answer", None)
                print("✅ LangChain response received")
            except Exception as e:
                print(f"❌ Error querying LangChain QA chain: {str(e)}")
                qa_response = None
        
        # Query LlamaIndex sheets query engine
        if sheets_query_engine:
            try:
                print("📊 Querying LlamaIndex sheets query engine...")
                sheets_result = sheets_query_engine.chat(prompt)
                sheets_response = sheets_result.response
                print("✅ LlamaIndex response received")
            except Exception as e:
                print(f"❌ Error querying LlamaIndex sheets query engine: {str(e)}")
                sheets_response = None
        
        # If we have both responses, combine them
        if qa_response and sheets_response:
            print("🔄 Combining responses from both systems...")
            # Initialize LLM for response combination
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3
            )
            combined_response = combine_responses_from_dual_systems(
                prompt, qa_response, sheets_response, llm
            )
            return combined_response
        
        # If we only have one response, return it
        elif qa_response:
            return qa_response
        elif sheets_response:
            return sheets_response
        else:
            return "I couldn't find relevant information in your documents or spreadsheets."
            
    except Exception as e:
        print(f"❌ Error in dual query system: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"

def main():
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("🤖 AI Document Assistant")
    st.markdown("Connect your Google Drive to process PDFs and Google Docs with semantic AI analysis")
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        st.markdown("### 🔐 Authentication Required")
        st.markdown("Please authenticate with Google to access your Drive documents.")
        
        # Google OAuth button
        if st.button("🔑 Connect Google Drive", type="primary"):
            auth_url = get_google_oauth_url()
            if auth_url:
                st.markdown(f"**Click the link below to authenticate:**")
                st.markdown(f"[🔗 Authenticate with Google]({auth_url})")
                st.info("After authentication, you'll be redirected back to this app.")
            else:
                st.error("Failed to generate authentication URL. Please check your Google OAuth configuration.")
        
        # Handle OAuth callback
        query_params = st.query_params
        if 'code' in query_params and not st.session_state.authenticated:
            st.info("🔄 Processing authentication...")
            
            # Extract the authorization code
            code = query_params['code']
            
            # Construct the callback URL properly
            callback_url = f"http://localhost:8501?code={code}"
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
                    st.success("✅ Successfully authenticated with Google!")
                    # Clear query params and redirect to clean URL
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error("❌ Authentication failed. Please try again.")
                    st.query_params.clear()
        
        st.markdown("---")
        st.markdown("### 📋 Setup Instructions")
        st.markdown("""
        1. **Google Cloud Setup**: Create a Google Cloud project and enable the Google Drive API
        2. **OAuth Configuration**: Set up OAuth 2.0 credentials for a web application
        3. **Environment Variables**: Add your Google OAuth credentials to your `.env` file:
           ```
           GOOGLE_OAUTH_CLIENT_ID=your_client_id
           GOOGLE_OAUTH_CLIENT_SECRET=your_client_secret
           GOOGLE_API_KEY=your_gemini_api_key
           ```
        4. **OpenSearch**: Ensure OpenSearch is running and accessible
        """)
        
        return
    
    # User is authenticated
    user_email = st.session_state.user_info.get('email', 'unknown')
    st.success(f"👋 Welcome, {user_email}!")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📁 Document Processing")
        
        # Force reprocess option
        force_reprocess = st.checkbox("🔄 Force Reprocess All Documents", 
                                    help="Clear existing data and reprocess all documents")
        
        if force_reprocess:
            if st.button("🗑️ Clear All Data", type="secondary"):
                # Clear processed files metadata
                metadata_path = get_user_metadata_path(user_email)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                # Clear OpenSearch index
                client = create_opensearch_client()
                if client and client.indices.exists(index=OPENSEARCH_INDEX):
                    client.indices.delete(index=OPENSEARCH_INDEX)
                
                # Clear sheets index if it exists
                if client and client.indices.exists(index="sheets_embeddings"):
                    client.indices.delete(index="sheets_embeddings")
                
                # Reset session state
                st.session_state.processed_pdfs = False
                st.session_state.processed_sheets = False
                st.session_state.vectorstore = None
                st.session_state.qa_chain = None
                st.session_state.sheets_query_engine = None
                st.session_state.start_chatting = False
                st.success("✅ All data cleared!")
                
                st.rerun()
        
        # Process documents button
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing your Google Drive..."):
                try:
                    process_all_user_documents(st.session_state.credentials, user_email)
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        # Show processing status
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
        
        # Show document count
        doc_count = get_document_count()
        if doc_count > 0:
            st.success(f"📊 {doc_count} documents in knowledge base")
        elif st.session_state.processed_pdfs:
            st.info("📊 No documents found in knowledge base")
        
        # Debug information (can be removed later)
        with st.expander("🔧 Debug Info"):
            st.write(f"processed_pdfs: {st.session_state.processed_pdfs}")
            st.write(f"processed_sheets: {st.session_state.processed_sheets}")
            st.write(f"start_chatting: {st.session_state.start_chatting}")
            st.write(f"has_qa_chain: {st.session_state.qa_chain is not None}")
            st.write(f"has_sheets_query_engine: {st.session_state.sheets_query_engine is not None}")
            st.write(f"has_vectorstore: {st.session_state.vectorstore is not None}")
            st.write(f"document_count: {doc_count}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This app processes your Google Drive documents using:
        
        **Document Processing (LangChain):**
        - **Semantic Chunking**: Advanced AI-powered text splitting
        - **Gemini Vision**: Image analysis and description
        - **OpenSearch**: Vector storage and retrieval
        
        **Sheets Processing (LlamaIndex):**
        - **LlamaParse**: Advanced spreadsheet parsing
        - **Dual-Query System**: Combines document and sheet analysis
        - **Intelligent Response Merging**: Seamless integration of multiple sources
        
        **Supports:** PDF files, Google Docs, Excel files, and Google Sheets
        """)
    
    # Main content area
    if not st.session_state.processed_pdfs and not st.session_state.processed_sheets and not st.session_state.start_chatting:
        # Check if there are existing documents in OpenSearch
        has_existing_docs = check_existing_documents()
        
        if has_existing_docs:
            st.info("📚 Found existing documents in your knowledge base!")
            st.markdown("You can start asking questions about your previously processed documents.")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Start Asking Questions", type="primary", use_container_width=True):
                    with st.spinner("Initializing chat interface..."):
                        if initialize_qa_chain_from_existing():
                            st.success("✅ Chat interface ready! You can now ask questions about your documents.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to initialize chat interface.")
            
            st.markdown("---")
            st.markdown("**Or** click 'Process Documents' in the sidebar to scan for new documents and sheets in your Google Drive.")
        else:
            st.info("📚 No documents or sheets processed yet. Click 'Process Documents' in the sidebar to get started.")
        return
    
    # Chat interface
    if st.session_state.processed_pdfs or st.session_state.start_chatting or st.session_state.processed_sheets:
        st.header("💬 Ask Questions About Your Documents & Sheets")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents and sheets..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response using dual-query system
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    # Check if we have either QA chain or sheets query engine
                    has_qa_chain = st.session_state.qa_chain is not None
                    has_sheets_engine = st.session_state.sheets_query_engine is not None
                    
                    if has_qa_chain or has_sheets_engine:
                        # Use dual-query system
                        answer = query_dual_systems(
                            prompt, 
                            st.session_state.qa_chain, 
                            st.session_state.sheets_query_engine
                        )
                        
                        # Display response
                        message_placeholder.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        message_placeholder.error("❌ No query engines initialized. Please process documents and sheets first.")
                except Exception as e:
                    message_placeholder.error(f"❌ Error generating response: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main() 
