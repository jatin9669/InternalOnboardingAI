import streamlit as st
import os
import pickle
import fitz  # PyMuPDF for PDF processing
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch
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
SHEETS_INDEX = 'sheets_embeddings'

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
if 'unified_retriever' not in st.session_state:
    st.session_state.unified_retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""
if 'start_chatting' not in st.session_state:
    st.session_state.start_chatting = False

print("🚀 Unified Document Assistant initialized")

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

def create_opensearch_index(client, index_name, dimension=768):
    """Create OpenSearch index with proper mapping for vector search."""
    try:
        # Delete existing index if it exists
        if client.indices.exists(index=index_name):
            print(f"🗑️ Deleting existing index: {index_name}")
            client.indices.delete(index=index_name)
        
        # Define the index mapping for vector search
        index_mapping = {
            "mappings": {
                "properties": {
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": dimension,  # Gemini embedding dimension
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
            index=index_name,
            body=index_mapping
        )
        
        print(f"✅ Created OpenSearch index: {index_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating OpenSearch index: {str(e)}")
        return False

def check_existing_documents(index_name=OPENSEARCH_INDEX):
    """Check if there are existing documents in OpenSearch."""
    try:
        client = create_opensearch_client()
        if not client:
            return False
        
        # Check if index exists
        if not client.indices.exists(index=index_name):
            return False
        
        # Count documents in the index
        count_response = client.count(index=index_name)
        document_count = count_response.get('count', 0)
        
        return document_count > 0
        
    except Exception as e:
        print(f"Error checking existing documents: {str(e)}")
        return False

def get_document_count(index_name=OPENSEARCH_INDEX):
    """Get the number of documents in OpenSearch."""
    try:
        print(f"🔍 Checking document count in OpenSearch index: {index_name}")
        client = create_opensearch_client()
        if not client:
            print("❌ Failed to create OpenSearch client")
            return 0
        
        # Check if index exists
        if not client.indices.exists(index=index_name):
            print(f"❌ Index {index_name} does not exist")
            return 0
        
        # Count documents in the index
        count_response = client.count(index=index_name)
        document_count = count_response.get('count', 0)
        print(f"📊 Found {document_count} documents in {index_name}")
        
        return document_count
        
    except Exception as e:
        print(f"❌ Error getting document count: {str(e)}")
        return 0

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
        return GoogleGenAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0.3)
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

class UnifiedRetriever:
    """Unified retriever that fetches relevant chunks from both PDFs/Docs and Sheets."""
    
    def __init__(self, pdf_vectorstore, sheets_index, embeddings):
        self.pdf_vectorstore = pdf_vectorstore
        self.sheets_index = sheets_index
        self.embeddings = embeddings
    
    def retrieve_relevant_chunks(self, query, k_pdfs=5, k_sheets=5, chat_history=None):
        """Retrieve relevant chunks from both PDFs/Docs and Sheets."""
        try:
            pdf_chunks = []
            sheets_chunks = []
            
            # Create context-aware query if history is available
            enhanced_query = query
            if chat_history and len(chat_history) > 0:
                # Use last few exchanges to create a more context-aware query
                recent_history = chat_history[-6:]  # Last 6 messages (3 exchanges)
                context_parts = []
                for message in recent_history:
                    if message["role"] == "user":
                        context_parts.append(f"User asked: {message['content']}")
                    else:
                        context_parts.append(f"Assistant responded: {message['content'][:100]}...")  # Truncate long responses
                
                context_summary = "\n".join(context_parts[-4:])  # Use last 4 context parts
                enhanced_query = f"Context from recent conversation:\n{context_summary}\n\nCurrent question: {query}"
                print(f"🔍 Using enhanced query with conversation context")
            
            # Retrieve from PDFs/Docs vectorstore
            if self.pdf_vectorstore:
                try:
                    pdf_docs = self.pdf_vectorstore.similarity_search(enhanced_query, k=k_pdfs)
                    pdf_chunks = [
                        {
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'source_type': 'pdf_doc'
                        }
                        for doc in pdf_docs
                    ]
                    print(f"📄 Retrieved {len(pdf_chunks)} PDF/Doc chunks")
                except Exception as e:
                    print(f"❌ Error retrieving PDF chunks: {str(e)}")
            
            # Retrieve from Sheets index
            if self.sheets_index:
                try:
                    # Use LlamaIndex query engine to get relevant chunks
                    query_engine = self.sheets_index.as_query_engine(
                        response_mode="tree_summarize",
                        similarity_top_k=k_sheets
                    )
                    sheets_response = query_engine.query(enhanced_query)
                    
                    # Extract source nodes (chunks) from response
                    if hasattr(sheets_response, 'source_nodes'):
                        sheets_chunks = [
                            {
                                'content': node.node.text,
                                'metadata': node.node.metadata,
                                'source_type': 'sheet'
                            }
                            for node in sheets_response.source_nodes
                        ]
                    print(f"📊 Retrieved {len(sheets_chunks)} sheet chunks")
                except Exception as e:
                    print(f"❌ Error retrieving sheet chunks: {str(e)}")
            
            return pdf_chunks, sheets_chunks
            
        except Exception as e:
            print(f"❌ Error in unified retrieval: {str(e)}")
            return [], []

def create_unified_response(pdf_chunks, sheets_chunks, query, llm, chat_history=None):
    """Create a unified response using all retrieved chunks in a single LLM call."""
    try:
        # Prepare context from PDF chunks
        pdf_context = ""
        if pdf_chunks:
            pdf_context = "**DOCUMENTS & PDFS:**\n"
            for i, chunk in enumerate(pdf_chunks, 1):
                source = chunk['metadata'].get('source', 'Unknown')
                page = chunk['metadata'].get('page', 'N/A')
                content_type = chunk['metadata'].get('type', 'text')
                pdf_context += f"\n--- Chunk {i} (Source: {source}, Page: {page}, Type: {content_type}) ---\n"
                pdf_context += chunk['content'] + "\n"
        
        # Prepare context from sheet chunks
        sheets_context = ""
        if sheets_chunks:
            sheets_context = "\n**SPREADSHEETS & EXCEL FILES:**\n"
            for i, chunk in enumerate(sheets_chunks, 1):
                source = chunk['metadata'].get('source', 'Unknown')
                sheets_context += f"\n--- Chunk {i} (Source: {source}) ---\n"
                sheets_context += chunk['content'] + "\n"
        
        # Prepare conversation history
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_context = "\n**CONVERSATION HISTORY:**\n"
            # Include last 5 exchanges for context (to avoid token limits)
            recent_history = chat_history[-10:]  # Last 10 messages (5 exchanges)
            for message in recent_history:
                role = "User" if message["role"] == "user" else "Assistant"
                history_context += f"{role}: {message['content']}\n"
        
        # Create unified prompt with strict guidelines
        unified_prompt = f"""# 🤖 UNIFIED AI DOCUMENT ASSISTANT

You are an advanced AI assistant with comprehensive access to information from both documents/PDFs and spreadsheets/Excel files. Your role is to provide accurate, well-structured, and contextually relevant responses based solely on the provided information.

## 📋 CORE RESPONSIBILITIES

### 🎯 Primary Objectives
- **Answer questions accurately** using only the provided context
- **Cite sources properly** by mentioning document names and content types
- **Maintain conversation flow** by referencing previous exchanges appropriately
- **Provide comprehensive insights** by combining text and visual information

### 🔍 Information Sources
- **Text Content**: Extracted from PDFs, Google Docs, and spreadsheets
- **Visual Content**: Image descriptions, diagrams, charts, and visual elements
- **Structured Data**: Numerical data, tables, and spreadsheet information
- **Metadata**: File sources, page numbers, and processing timestamps

## 📊 CONTEXT INFORMATION

### 📄 DOCUMENTS & PDFS
{pdf_context}

### 📈 SPREADSHEETS & EXCEL FILES
{sheets_context}

### 💬 CONVERSATION HISTORY
{history_context}

## 🎨 VISUAL CONTENT GUIDELINES

### 🔍 Image-Related Keywords to Monitor
**Visual Elements:**
- image, picture, photo, screenshot, diagram, chart, graph, logo, icon

**Descriptive Terms:**
- visual, appearance, look like, show, display, depict, illustrate

**Interface Elements:**
- dashboard, interface, UI, layout, design, mockup, wireframe

**Technical Specifications:**
- color, style, format, size, dimensions, resolution

### 📸 Image Processing Protocol
1. **Priority Check**: When image-related keywords are detected, prioritize image descriptions
2. **Comprehensive Analysis**: Use both text and visual information for complete answers
3. **Source Attribution**: Always mention whether information came from text or images
4. **Visual Description**: When images are referenced, describe their content based on available descriptions
5. **Fallback Handling**: If visual content is requested but not found, clearly state this

## 🔗 URL AND LINK HANDLING

### 📎 Link Management
- **Always include URLs** when present in the context
- **Format as clickable links** using proper markdown syntax
- **Reference both section names and links** when discussing specific content
- **Maintain link integrity** throughout the response

## 📝 RESPONSE STRUCTURE GUIDELINES

### 🎯 Answer Format
1. **Direct Response**: Provide a clear, concise answer to the user's question
2. **Source Citation**: Reference document names and content types (text/image)
3. **Context Integration**: Use conversation history for follow-up questions
4. **Data Presentation**: Use bullet points or numbered lists for multiple items
5. **Professional Tone**: Maintain helpful and informative communication style

### 📊 Data Analysis Approach
- **Numerical Context**: Provide trends and context for numerical data
- **Insight Generation**: Offer analysis and explanations, not just raw data
- **Cell References**: Clearly reference specific cells, rows, or columns when applicable
- **Trend Identification**: Highlight patterns and relationships in the data

## ⚠️ STRICT COMPLIANCE RULES

### 🚫 Prohibited Actions
- **Never answer from personal knowledge** - use only provided context
- **No speculation** about information not present in the context
- **No repetition** of previously discussed information unless specifically requested
- **No inclusion** of irrelevant details from the chunks

### ✅ Required Actions
- **Eliminate duplicates** and redundant information
- **Be concise and precise** in all responses
- **Cite sources exactly once** for each piece of information
- **Clearly state** when information is not available in the context

## 🎯 CURRENT QUERY

**USER QUESTION:** {query}

## 📤 RESPONSE GENERATION

Based on the comprehensive context provided above, generate a well-structured response that:

1. **Directly addresses** the user's question
2. **Integrates information** from all relevant sources (text, images, spreadsheets)
3. **Maintains conversation continuity** using the provided history
4. **Follows all formatting and citation guidelines**
5. **Provides actionable insights** when applicable

**ANSWER:**"""

        # Get response from LLM
        response = llm.invoke(unified_prompt)
        return response.content
        
    except Exception as e:
        print(f"❌ Error creating unified response: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"

def process_all_user_documents_unified(credentials, user_email):
    """Process all PDF, Google Doc, and Sheet files using the unified approach."""
    try:
        # Create OpenSearch client
        client = create_opensearch_client()
        if not client:
            raise Exception("Failed to create OpenSearch client")

        # Check if indices exist, if not create them
        if not client.indices.exists(index=OPENSEARCH_INDEX):
            create_opensearch_index(client, OPENSEARCH_INDEX)
        
        if not client.indices.exists(index=SHEETS_INDEX):
            create_opensearch_index(client, SHEETS_INDEX)

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
                
                # Initialize unified retriever from existing documents
                if initialize_unified_retriever_from_existing():
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
        all_sheet_documents = []
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
            
            # Create embeddings using Gemini
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            
            # Apply semantic chunking to documents
            text_splits = []
            for doc in documents:
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
            
            if text_splits:
                st.info(f"📊 Created {len(text_splits)} semantic chunks from {len(documents)} documents")
                
                # Create vector store for PDFs/Docs
                pdf_vectorstore = OpenSearchVectorSearch(
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
                print(f"Adding {len(text_splits)} documents to PDF vector store")
                pdf_vectorstore.add_documents(text_splits)
            else:
                print("⚠️ No document chunks created")
                pdf_vectorstore = None
        else:
            pdf_vectorstore = None
        
        # Process sheets with LlamaIndex
        sheets_index = None
        if all_sheet_documents:
            st.info("📊 Processing sheets with LlamaIndex...")
            
            try:
                sheets_index = create_sheets_index(all_sheet_documents)
                if sheets_index:
                    st.session_state.processed_sheets = True
                    st.success("✅ Sheets index created successfully!")
                else:
                    st.error("❌ Failed to create sheets index")
            except Exception as e:
                st.error(f"❌ Error creating sheets index: {str(e)}")
        
        # Create unified retriever
        if pdf_vectorstore or sheets_index:
            try:
                # Initialize LLM
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.3
                )
                
                # Create unified retriever
                unified_retriever = UnifiedRetriever(pdf_vectorstore, sheets_index, embeddings)
                
                # Update session state
                st.session_state.unified_retriever = unified_retriever
                st.session_state.llm = llm
                st.session_state.processed_pdfs = True
                st.session_state.start_chatting = True
                
                st.success("✅ Unified retriever created successfully!")
                
            except Exception as e:
                st.error(f"❌ Error creating unified retriever: {str(e)}")
        
        # Save updated metadata
        save_processed_files_metadata(processed_files, user_email)
        
        # Update session state
        st.session_state.processing_status = "✅ Processing complete! You can now ask questions about your documents and sheets."
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        st.session_state.processing_status = f"❌ Error: {str(e)}"

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

def create_sheets_index(all_documents):
    """Create and return the LlamaIndex index for sheets processing."""
    try:
        print("🔧 Initializing LlamaIndex components for sheets...")
        
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
        sheets_idx = SHEETS_INDEX
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
        
        print("✅ Sheets index created successfully")
        return index
        
    except Exception as e:
        print(f"❌ Error creating sheets index: {str(e)}")
        print("Detailed error:")
        print(traceback.format_exc())
        return None

def initialize_unified_retriever_from_existing():
    """Initialize unified retriever from existing OpenSearch documents."""
    try:
        print("🔧 Starting unified retriever initialization...")
        
        # Create embeddings
        print("📝 Creating embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Create vector store connection to existing index
        print("🔗 Connecting to OpenSearch vector store...")
        pdf_vectorstore = OpenSearchVectorSearch(
            embedding_function=embeddings,
            opensearch_url=OPENSEARCH_URL,
            index_name=OPENSEARCH_INDEX,
            vector_field="vector_field",
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD) if OPENSEARCH_USERNAME else None,
            use_ssl=OPENSEARCH_URL.startswith('https'),
            verify_certs=False,
            engine="lucene"
        )
        
        # Check if sheets index exists
        sheets_index = None
        if check_existing_documents(SHEETS_INDEX):
            print("📊 Found existing sheets index, connecting...")
            try:
                # Initialize LlamaIndex components
                llm = initialize_llamaindex_llm()
                sheets_embeddings = initialize_llamaindex_embeddings()
                
                if llm and sheets_embeddings:
                    Settings.llm = llm
                    Settings.embed_model = sheets_embeddings
                    
                    # Create vector store connection
                    vector_client = OpensearchVectorClient(
                        OPENSEARCH_URL,
                        SHEETS_INDEX,
                        768,
                        embedding_field="embedding",
                        text_field="content"
                    )
                    vector_store = OpensearchVectorStore(vector_client)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    
                    # Load existing index
                    sheets_index = VectorStoreIndex.from_vector_store(
                        vector_store=vector_store,
                        embed_model=sheets_embeddings
                    )
                    print("✅ Connected to existing sheets index")
            except Exception as e:
                print(f"⚠️ Could not connect to sheets index: {str(e)}")
        
        # Initialize LLM
        print("🤖 Initializing LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
        
        # Create unified retriever
        print("🔗 Creating unified retriever...")
        unified_retriever = UnifiedRetriever(pdf_vectorstore, sheets_index, embeddings)
        
        # Update session state
        print("💾 Updating session state...")
        st.session_state.unified_retriever = unified_retriever
        st.session_state.llm = llm
        st.session_state.start_chatting = True
        
        print("✅ Unified retriever initialization completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing unified retriever: {str(e)}")
        st.error(f"Error initializing unified retriever: {str(e)}")
        return False

def query_unified_system(prompt, unified_retriever, llm, chat_history=None):
    """Query the unified system using the new approach."""
    try:
        print("🔍 Querying unified system...")
        
        # Retrieve relevant chunks from both sources with history context
        pdf_chunks, sheets_chunks = unified_retriever.retrieve_relevant_chunks(prompt, chat_history=chat_history)
        
        print(f"📄 Retrieved {len(pdf_chunks)} PDF chunks and {len(sheets_chunks)} sheet chunks")
        
        # Create unified response in a single LLM call with history
        response = create_unified_response(pdf_chunks, sheets_chunks, prompt, llm, chat_history)
        
        print("✅ Unified response generated successfully")
        return response
        
    except Exception as e:
        print(f"❌ Error in unified query system: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"

def main():
    st.set_page_config(
        page_title="Unified AI Document Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Unified AI Document Assistant")
    st.markdown("Connect your Google Drive to process PDFs, Google Docs, and Sheets with unified AI analysis")
    
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
           LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
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
                
                # Clear OpenSearch indices
                client = create_opensearch_client()
                if client:
                    if client.indices.exists(index=OPENSEARCH_INDEX):
                        client.indices.delete(index=OPENSEARCH_INDEX)
                    if client.indices.exists(index=SHEETS_INDEX):
                        client.indices.delete(index=SHEETS_INDEX)
                
                # Reset session state
                st.session_state.processed_pdfs = False
                st.session_state.processed_sheets = False
                st.session_state.unified_retriever = None
                st.session_state.llm = None
                st.session_state.start_chatting = False
                st.success("✅ All data cleared!")
                
                st.rerun()
        
        # Process documents button
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing your Google Drive..."):
                try:
                    process_all_user_documents_unified(st.session_state.credentials, user_email)
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        # Show processing status
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
        
        # Show document counts
        pdf_count = get_document_count(OPENSEARCH_INDEX)
        sheets_count = get_document_count(SHEETS_INDEX)
        
        if pdf_count > 0:
            st.success(f"📄 {pdf_count} PDF/Doc chunks in knowledge base")
        if sheets_count > 0:
            st.success(f"📊 {sheets_count} sheet chunks in knowledge base")
        
        # Debug information
        with st.expander("🔧 Debug Info"):
            st.write(f"processed_pdfs: {st.session_state.processed_pdfs}")
            st.write(f"processed_sheets: {st.session_state.processed_sheets}")
            st.write(f"start_chatting: {st.session_state.start_chatting}")
            st.write(f"has_unified_retriever: {st.session_state.unified_retriever is not None}")
            st.write(f"has_llm: {st.session_state.llm is not None}")
            st.write(f"pdf_chunks: {pdf_count}")
            st.write(f"sheet_chunks: {sheets_count}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        **Unified Approach:**
        - **Single LLM Call**: Retrieves chunks from both sources and generates response in one call
        - **Better Integration**: Sees all relevant information simultaneously
        - **More Efficient**: Fewer API calls, lower latency, lower cost
        - **Better Quality**: More coherent responses with better cross-references
        
        **Supports:** PDF files, Google Docs, Excel files, and Google Sheets
        """)
    
    # Main content area
    if not st.session_state.processed_pdfs and not st.session_state.processed_sheets and not st.session_state.start_chatting:
        # Check if there are existing documents in OpenSearch
        has_existing_docs = check_existing_documents(OPENSEARCH_INDEX) or check_existing_documents(SHEETS_INDEX)
        
        if has_existing_docs:
            st.info("📚 Found existing documents in your knowledge base!")
            st.markdown("You can start asking questions about your previously processed documents.")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Start Asking Questions", type="primary", use_container_width=True):
                    with st.spinner("Initializing unified chat interface..."):
                        if initialize_unified_retriever_from_existing():
                            st.success("✅ Unified chat interface ready! You can now ask questions about your documents and sheets.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to initialize unified chat interface.")
            
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
            
            # Generate response using unified system
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    # Check if we have unified retriever and LLM
                    has_unified_retriever = st.session_state.unified_retriever is not None
                    has_llm = st.session_state.llm is not None
                    
                    if has_unified_retriever and has_llm:
                        # Use unified query system
                        answer = query_unified_system(
                            prompt, 
                            st.session_state.unified_retriever, 
                            st.session_state.llm,
                            st.session_state.messages
                        )
                        
                        # Display response
                        message_placeholder.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        message_placeholder.error("❌ Unified retriever not initialized. Please process documents and sheets first.")
                except Exception as e:
                    message_placeholder.error(f"❌ Error generating response: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main() 
