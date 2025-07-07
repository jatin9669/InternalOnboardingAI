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
from googleapiclient.errors import HttpError

# Import spaCy metadata handler
from spacy_metadata_handler import get_spacy_metadata_handler, QueryClassification

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
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.activity',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.activity.readonly',
    'https://www.googleapis.com/auth/drive.metadata',
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

# Initialize spaCy metadata handler
try:
    print("🤖 Initializing spaCy metadata handler...")
    spacy_handler = get_spacy_metadata_handler()
    
    # Create metadata index in OpenSearch
    if spacy_handler.opensearch_client:
        success = spacy_handler.create_metadata_index()
        if success:
            print("✅ spaCy metadata index created successfully")
        else:
            print("⚠️ Failed to create spaCy metadata index")
    else:
        print("⚠️ OpenSearch client not available for spaCy metadata")
        
except Exception as e:
    print(f"⚠️ Error initializing spaCy metadata handler: {str(e)}")

def classify_query(query):
    """
    Classify a query as either 'content' or 'metadata' using spaCy-based approach.
    
    Args:
        query (str): The user's query
        
    Returns:
        dict: Classification result with type and confidence
    """
    try:
        # Use spaCy-based classification
        spacy_handler = get_spacy_metadata_handler()
        classification = spacy_handler.classify_query(query)
        
        # Convert to the expected format
        return {
            "classification": classification.classification,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning,
            "query_type": classification.query_type
        }
        
    except Exception as e:
        print(f"Error in spaCy query classification: {str(e)}")
        # Fallback to original LLM-based classification
        return fallback_query_classification(query)

def fallback_query_classification(query):
    """
    Fallback classification using keyword matching when LLM classification fails.
    """
    query_lower = query.lower()
    
    # Metadata keywords (queries about file properties, structure, etc.)
    metadata_keywords = [
        'how many', 'count', 'number of', 'file type', 'file size', 'file name', 'file format',
        'when', 'date', 'time', 'modified', 'created', 'processed',
        'recent', 'oldest', 'newest', 'largest', 'smallest',
        'show me all', 'list', 'what files', 'which files',
        'file properties', 'metadata', 'structure', 'organization',
        'file count', 'document count', 'total files', 'file list'
    ]
    
    # Content keywords (queries about actual content, data, information)
    content_keywords = [
        'what does', 'what is', 'what are', 'find', 'search',
        'information about', 'data about', 'content', 'says',
        'explain', 'describe', 'analyze', 'summarize', 'key points',
        'topics', 'subjects', 'details', 'facts', 'information',
        'containing', 'with', 'about', 'reviews', 'products', 'companies',
        'data', 'table', 'sheet', 'spreadsheet'
    ]
    
    metadata_score = sum(1 for keyword in metadata_keywords if keyword in query_lower)
    content_score = sum(1 for keyword in content_keywords if keyword in query_lower)
    
    # Special handling for queries that ask about content within files
    if 'containing' in query_lower or 'with' in query_lower:
        content_score += 2  # Boost content score for "containing" queries
    
    # Special handling for specific content types
    if any(word in query_lower for word in ['reviews', 'products', 'companies', 'data', 'table']):
        content_score += 1  # Boost content score for specific content types
    
    if metadata_score > content_score:
        return {
            "classification": "metadata",
            "confidence": 0.7,
            "reasoning": "Keyword-based classification: query contains metadata-related terms"
        }
    else:
        return {
            "classification": "content",
            "confidence": 0.7,
            "reasoning": "Keyword-based classification: query appears to ask about content"
        }

def get_user_enhanced_metadata_path(user_email):
    """Get user-specific enhanced metadata path for comprehensive file information."""
    user_hash = hashlib.md5(user_email.encode()).hexdigest()
    return f"vectorstores/user_{user_hash}_enhanced_metadata.json"

def save_enhanced_file_metadata(file_metadata, user_email):
    """Save comprehensive metadata about processed files to both JSON and OpenSearch."""
    # Save to JSON file
    metadata_path = get_user_enhanced_metadata_path(user_email)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # Load existing metadata if it exists
    existing_metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
        except Exception as e:
            print(f"Failed to load existing metadata: {e}")
    
    # Update with new metadata
    existing_metadata.update(file_metadata)
    
    # Save updated metadata to JSON
    with open(metadata_path, 'w') as f:
        json.dump(existing_metadata, f, indent=2)
    
    print(f"✅ Enhanced metadata saved to {metadata_path}")
    
    # Also save to OpenSearch using spaCy handler
    try:
        spacy_handler = get_spacy_metadata_handler()
        success = spacy_handler.store_metadata_in_opensearch(file_metadata, user_email)
        if success:
            print(f"✅ Metadata also stored in OpenSearch for user {user_email}")
        else:
            print(f"⚠️ Failed to store metadata in OpenSearch for user {user_email}")
    except Exception as e:
        print(f"⚠️ Error storing metadata in OpenSearch: {str(e)}")

def load_enhanced_file_metadata(user_email):
    """Load comprehensive metadata about processed files."""
    metadata_path = get_user_enhanced_metadata_path(user_email)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load enhanced metadata: {e}")
            return {}
    return {}

def delete_file_from_enhanced_metadata(file_name, user_email):
    """Delete a file entry from the enhanced metadata JSON file."""
    try:
        metadata_path = get_user_enhanced_metadata_path(user_email)
        if not os.path.exists(metadata_path):
            return False
        
        # Load existing metadata
        with open(metadata_path, 'r') as f:
            enhanced_metadata = json.load(f)
        
        # Find and delete the file entry
        file_hash_to_delete = None
        for file_hash, metadata in enhanced_metadata.items():
            if metadata.get('file_name') == file_name:
                file_hash_to_delete = file_hash
                break
        
        if file_hash_to_delete:
            del enhanced_metadata[file_hash_to_delete]
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2)
            
            print(f"✅ Deleted {file_name} from enhanced metadata")
            return True
        else:
            print(f"⚠️ File {file_name} not found in enhanced metadata")
            return False
            
    except Exception as e:
        print(f"❌ Error deleting file from enhanced metadata: {str(e)}")
        return False

def delete_chunks_from_opensearch_by_filename(file_name, index_name=OPENSEARCH_INDEX):
    """Delete all chunks from OpenSearch index that belong to a specific file."""
    try:
        client = create_opensearch_client()
        if not client:
            print("❌ Failed to create OpenSearch client")
            return False
        
        # Check if index exists
        if not client.indices.exists(index=index_name):
            print(f"❌ Index {index_name} does not exist")
            return False
        
        # Delete documents by file name in metadata
        delete_query = {
            "query": {
                "match": {
                    "metadata.source": file_name
                }
            }
        }
        
        response = client.delete_by_query(
            index=index_name,
            body=delete_query
        )
        
        deleted_count = response.get('deleted', 0)
        print(f"✅ Deleted {deleted_count} chunks for file '{file_name}' from {index_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting chunks from OpenSearch: {str(e)}")
        return False

def delete_metadata_from_opensearch_by_filename(file_name, user_email):
    """Delete metadata from OpenSearch metadata index for a specific file."""
    try:
        # Get spaCy metadata handler
        spacy_handler = get_spacy_metadata_handler()
        if not spacy_handler or not spacy_handler.opensearch_client:
            print("❌ spaCy metadata handler or OpenSearch client not available")
            return False
        
        client = spacy_handler.opensearch_client
        
        # Check if metadata index exists
        if not client.indices.exists(index="file_metadata"):
            print("❌ Metadata index does not exist")
            return False
        
        # Delete documents by file name and user email
        delete_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_email": user_email}},
                        {"match": {"file_name": file_name}}
                    ]
                }
            }
        }
        
        response = client.delete_by_query(
            index="file_metadata",
            body=delete_query
        )
        
        deleted_count = response.get('deleted', 0)
        print(f"✅ Deleted {deleted_count} metadata entries for file '{file_name}' from metadata index")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting metadata from OpenSearch: {str(e)}")
        return False

def delete_metadata_from_opensearch_by_file_id(file_id, user_email):
    """Delete metadata from OpenSearch metadata index for a specific file by file_id."""
    try:
        # Get spaCy metadata handler
        spacy_handler = get_spacy_metadata_handler()
        if not spacy_handler or not spacy_handler.opensearch_client:
            print("❌ spaCy metadata handler or OpenSearch client not available")
            return False
        
        client = spacy_handler.opensearch_client
        
        # Check if metadata index exists
        if not client.indices.exists(index="file_metadata"):
            print("❌ Metadata index does not exist")
            return False
        
        # Delete documents by file_id and user email
        delete_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_email": user_email}},
                        {"term": {"file_id": file_id}}
                    ]
                }
            }
        }
        
        response = client.delete_by_query(
            index="file_metadata",
            body=delete_query
        )
        
        deleted_count = response.get('deleted', 0)
        print(f"✅ Deleted {deleted_count} metadata entries for file_id '{file_id}' from metadata index")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting metadata from OpenSearch: {str(e)}")
        return False

def check_file_modification(file_info, user_email):
    """
    Check if a file has been modified since last processing.
    
    Args:
        file_info (dict): Current file information from Google Drive
        user_email (str): User's email
        
    Returns:
        tuple: (is_modified, existing_metadata) where is_modified is bool and existing_metadata is dict or None
    """
    try:
        file_id = file_info['id']
        file_name = file_info['name']
        current_modified_time = file_info.get('modifiedTime', '')
        
        # Load enhanced metadata
        enhanced_metadata = load_enhanced_file_metadata(user_email)
        
        # Find existing metadata for this file by file_id
        existing_metadata = None
        existing_file_hash = None
        
        for file_hash, metadata in enhanced_metadata.items():
            if metadata.get('file_id') == file_id:
                existing_metadata = metadata
                existing_file_hash = file_hash
                break
        
        if existing_metadata:
            stored_modified_time = existing_metadata.get('modified_time', '')
            
            # Compare modification times
            if current_modified_time != stored_modified_time:
                print(f"🔄 File '{file_name}' has been modified:")
                print(f"   Old modified time: {stored_modified_time}")
                print(f"   New modified time: {current_modified_time}")
                return True, existing_metadata
            else:
                print(f"✅ File '{file_name}' is up to date")
                return False, existing_metadata
        else:
            print(f"🆕 File '{file_name}' is new (not previously processed)")
            return False, None
            
    except Exception as e:
        print(f"❌ Error checking file modification: {str(e)}")
        return False, None

def cleanup_modified_file(file_name, user_email, file_id=None):
    """
    Clean up a modified file by deleting its chunks from vector databases and metadata.
    
    Args:
        file_name (str): Name of the file to cleanup
        user_email (str): User's email
        file_id (str, optional): File ID for more precise metadata cleanup
        
    Returns:
        bool: True if cleanup was successful
    """
    try:
        print(f"🧹 Cleaning up modified file: {file_name}")
        
        # Delete chunks from PDF/Docs index
        pdf_cleanup_success = delete_chunks_from_opensearch_by_filename(file_name, OPENSEARCH_INDEX)
        
        # Delete chunks from sheets index
        sheets_cleanup_success = delete_chunks_from_opensearch_by_filename(file_name, SHEETS_INDEX)
        
        # Delete from enhanced metadata JSON file
        enhanced_metadata_cleanup_success = delete_file_from_enhanced_metadata(file_name, user_email)
        
        # Delete from OpenSearch metadata index
        metadata_index_cleanup_success = False
        if file_id:
            # Use file_id for more precise cleanup
            metadata_index_cleanup_success = delete_metadata_from_opensearch_by_file_id(file_id, user_email)
        else:
            # Fallback to filename-based cleanup
            metadata_index_cleanup_success = delete_metadata_from_opensearch_by_filename(file_name, user_email)
        
        if pdf_cleanup_success or sheets_cleanup_success or enhanced_metadata_cleanup_success or metadata_index_cleanup_success:
            print(f"✅ Cleanup completed for {file_name}")
            return True
        else:
            print(f"⚠️ No cleanup actions were performed for {file_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error during file cleanup: {str(e)}")
        return False

def extract_file_metadata(file_info, file_type, additional_info=None):
    """
    Extract comprehensive metadata from a file.
    
    Args:
        file_info (dict): Basic file information from Google Drive
        file_type (str): Type of file (pdf, google_doc, sheet)
        additional_info (dict): Additional processing information
        
    Returns:
        dict: Comprehensive file metadata
    """
    metadata = {
        "file_id": file_info['id'],
        "file_name": file_info['name'],
        "file_type": file_type,
        "mime_type": file_info.get('mimeType', ''),
        "size": file_info.get('size', '0'),
        "created_time": file_info.get('createdTime', ''),
        "modified_time": file_info.get('modifiedTime', ''),
        "last_opened_time": file_info.get('viewedByMeTime', ''),
        "processed_time": datetime.now().isoformat(),
        "owners": file_info.get('owners', []),
        "permissions": file_info.get('permissions', []),
        "web_view_link": file_info.get('webViewLink', ''),
        "web_content_link": file_info.get('webContentLink', ''),
        "thumbnail_link": file_info.get('thumbnailLink', ''),
        "parents": file_info.get('parents', []),
        "trashed": file_info.get('trashed', False),
        "starred": file_info.get('starred', False),
        "shared": file_info.get('shared', False),
        "viewed_by_me_time": file_info.get('viewedByMeTime', ''),
        "viewed_by_me": file_info.get('viewedByMe', False),
        "capabilities": file_info.get('capabilities', {}),
        "export_links": file_info.get('exportLinks', {}),
        "app_properties": file_info.get('appProperties', {}),
        "properties": file_info.get('properties', {}),
        "image_media_metadata": file_info.get('imageMediaMetadata', {}),
        "video_media_metadata": file_info.get('videoMediaMetadata', {}),
        "processing_status": "completed",
        "processing_errors": [],
        # Add revision information fields
        "revisions": [],
        "revision_count": 0
    }
    
    # Add file-type specific metadata
    if file_type == 'pdf':
        metadata.update({
            "page_count": additional_info.get('page_count', 0),
            "text_chunks": additional_info.get('text_chunks', 0),
            "image_chunks": additional_info.get('image_chunks', 0),
            "total_chunks": additional_info.get('total_chunks', 0),
            "extracted_links": additional_info.get('extracted_links', []),
            "has_images": additional_info.get('has_images', False),
            "file_size_mb": round(int(metadata['size']) / (1024 * 1024), 2) if metadata['size'] != '0' else 0
        })
    elif file_type == 'google_doc':
        metadata.update({
            "word_count": additional_info.get('word_count', 0),
            "character_count": additional_info.get('character_count', 0),
            "text_chunks": additional_info.get('text_chunks', 0),
            "total_chunks": additional_info.get('text_chunks', 0),
            "file_size_mb": round(int(metadata['size']) / (1024 * 1024), 2) if metadata['size'] != '0' else 0
        })
    elif file_type == 'sheet':
        metadata.update({
            "sheet_count": additional_info.get('sheet_count', 0),
            "row_count": additional_info.get('row_count', 0),
            "column_count": additional_info.get('column_count', 0),
            "text_chunks": additional_info.get('text_chunks', 0),
            "total_chunks": additional_info.get('text_chunks', 0),
            "file_size_mb": round(int(metadata['size']) / (1024 * 1024), 2) if metadata['size'] != '0' else 0
        })
    
    return metadata

def handle_metadata_query(query, user_email):
    """
    Handle metadata-based queries using spaCy and OpenSearch.
    
    Args:
        query (str): The metadata query
        user_email (str): User's email for accessing their metadata
        
    Returns:
        str: Formatted response based on metadata
    """
    try:
        # Use spaCy-based metadata handler
        spacy_handler = get_spacy_metadata_handler()
        
        # Handle the metadata query using spaCy and OpenSearch
        response = spacy_handler.handle_metadata_query(query, user_email)
        
        return response
        
    except Exception as e:
        print(f"Error handling metadata query: {str(e)}")
        return f"❌ Error processing metadata query: {str(e)}"

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
                fields='nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, viewedByMeTime, owners, permissions, webViewLink, webContentLink, thumbnailLink, parents, trashed, starred, shared, viewedByMe, capabilities, exportLinks, appProperties, properties, imageMediaMetadata, videoMediaMetadata)',
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
                fields='nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, viewedByMeTime, owners, permissions, webViewLink, webContentLink, thumbnailLink, parents, trashed, starred, shared, viewedByMe, capabilities, exportLinks, appProperties, properties, imageMediaMetadata, videoMediaMetadata)',
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
                fields='nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, viewedByMeTime, owners, permissions, webViewLink, webContentLink, thumbnailLink, parents, trashed, starred, shared, viewedByMe, capabilities, exportLinks, appProperties, properties, imageMediaMetadata, videoMediaMetadata)',
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
                fields='nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, viewedByMeTime, owners, permissions, webViewLink, webContentLink, thumbnailLink, parents, trashed, starred, shared, viewedByMe, capabilities, exportLinks, appProperties, properties, imageMediaMetadata, videoMediaMetadata)',
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

def download_google_doc_content(service, file_id, file_name, file_info=None):
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
        
        # Extract comprehensive metadata for storage
        if file_info:
            # Count words and characters
            word_count = len(content.split())
            character_count = len(content)
            
            additional_info = {
                "word_count": word_count,
                "character_count": character_count,
                "text_chunks": 1,  # Google Docs are typically single chunks
                "total_chunks": 1
            }
            
            # Store enhanced metadata
            file_hash = hashlib.md5(f"{file_id}_{file_info.get('modifiedTime', '')}".encode()).hexdigest()
            enhanced_metadata = extract_file_metadata(file_info, 'google_doc', additional_info)
            
            # Get user email from session state
            user_email = st.session_state.user_info.get('email', 'unknown') if hasattr(st, 'session_state') and st.session_state.user_info else 'unknown'
            
            # Save enhanced metadata
            save_enhanced_file_metadata({file_hash: enhanced_metadata}, user_email)
        
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

def process_pdf_with_images(pdf_path, file_name, file_id, file_info=None):
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
        all_links = []
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
                all_links.extend(page["links"])
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
        
        # Extract comprehensive metadata for storage
        if file_info:
            additional_info = {
                "page_count": len(extracted_pages),
                "text_chunks": text_doc_count,
                "image_chunks": image_doc_count,
                "total_chunks": len(documents),
                "extracted_links": list(set(all_links)),  # Remove duplicates
                "has_images": image_doc_count > 0
            }
            
            # Store enhanced metadata
            file_hash = hashlib.md5(f"{file_id}_{file_info.get('modifiedTime', '')}".encode()).hexdigest()
            enhanced_metadata = extract_file_metadata(file_info, 'pdf', additional_info)
            
            # Get user email from session state
            user_email = st.session_state.user_info.get('email', 'unknown') if hasattr(st, 'session_state') and st.session_state.user_info else 'unknown'
            
            # Save enhanced metadata
            save_enhanced_file_metadata({file_hash: enhanced_metadata}, user_email)
        
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
    
    def retrieve_relevant_chunks(self, query, k_pdfs=20, k_sheets=10, chat_history=None):
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
            
            # Detect if this is a sheet-specific query (generic approach)
            sheet_keywords = ['sheet', 'spreadsheet', 'excel', 'table', 'data']
            is_sheet_query = any(keyword in query.lower() for keyword in sheet_keywords)
            
            # Adjust retrieval parameters based on query type
            if is_sheet_query:
                print("📊 Detected sheet-specific query, prioritizing sheet retrieval...")
                k_sheets = max(k_sheets, 10)  # Increase sheet retrieval
                k_pdfs = min(k_pdfs, 5)  # Reduce PDF retrieval for sheet queries
            
            # Retrieve from PDFs/Docs vectorstore
            if self.pdf_vectorstore:
                try:
                    # Use hybrid search approach to avoid dominance by single documents
                    print("🔍 Using hybrid search approach...")
                    client = create_opensearch_client()
                    
                    # Extract key terms from the query
                    query_terms = enhanced_query.lower().split()
                    meaningful_terms = [term for term in query_terms if len(term) > 3 and term not in ['the', 'and', 'for', 'with', 'this', 'that', 'file', 'containing', 'has', 'includes', 'what', 'does', 'tell', 'me', 'about']]
                    
                    # Look for potential file names in the query
                    potential_file_names = []
                    for i, term in enumerate(query_terms):
                        if term in ['pdf', 'document', 'file'] and i + 1 < len(query_terms):
                            potential_file_names.append(query_terms[i + 1])
                        elif term.endswith('.pdf') or term.endswith('.doc'):
                            potential_file_names.append(term)
                    
                    # Extract potential document names from query using intelligent parsing
                    # Look for patterns that might indicate document names
                    query_words = enhanced_query.lower().split()
                    
                    # Check for file extensions
                    for word in query_words:
                        if word.endswith('.pdf') or word.endswith('.doc') or word.endswith('.docx'):
                            potential_file_names.append(word)
                    
                    # Check for quoted strings (potential file names)
                    import re
                    quoted_names = re.findall(r'"([^"]+)"', enhanced_query)
                    potential_file_names.extend(quoted_names)
                    
                    # Look for capitalized phrases that might be document names
                    # Split by common separators and look for title-case patterns
                    title_patterns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', enhanced_query)
                    for pattern in title_patterns:
                        if len(pattern.split()) >= 2:  # At least 2 words for a document name
                            potential_file_names.append(pattern)
                    
                    # Remove duplicates and filter out common words
                    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
                    potential_file_names = [name for name in potential_file_names if name.lower() not in common_words]
                    potential_file_names = list(set(potential_file_names))  # Remove duplicates
                    
                    print(f"🎯 Potential file names to search: {potential_file_names}")
                    
                    # Build hybrid search query
                    search_queries = []
                    
                    # 1. Semantic search with query text
                    search_queries.append({
                        'match': {
                            'text': {
                                'query': enhanced_query,
                                'boost': 1.0
                            }
                        }
                    })
                    
                    # 2. Phrase matching for exact terms
                    search_queries.append({
                        'match_phrase': {
                            'text': {
                                'query': enhanced_query,
                                'boost': 2.0
                            }
                        }
                    })
                    
                    # 3. Boost specific file names if mentioned (using fuzzy matching)
                    for file_name in potential_file_names:
                        search_queries.append({
                            'fuzzy': {
                                'metadata.source': {
                                    'value': file_name,
                                    'fuzziness': 'AUTO',
                                    'boost': 5.0  # High boost for file name matches
                                }
                            }
                        })
                        search_queries.append({
                            'match_phrase': {
                                'metadata.source': {
                                    'query': file_name,
                                    'boost': 10.0  # Very high boost for exact file name matches
                                }
                            }
                        })
                    
                    # 4. Add fuzzy matching for document names in content
                    for file_name in potential_file_names:
                        search_queries.append({
                            'fuzzy': {
                                'text': {
                                    'value': file_name,
                                    'fuzziness': 'AUTO',
                                    'boost': 3.0
                                }
                            }
                        })
                    
                    # 5. Add meaningful term searches
                    for term in meaningful_terms:
                        search_queries.append({
                            'match': {
                                'text': {
                                    'query': term,
                                    'boost': 1.5
                                }
                            }
                        })
                    
                    # 6. Add k-NN vector search for semantic similarity
                    embedding = self.embeddings.embed_query(enhanced_query)
                    search_queries.append({
                        'knn': {
                            'vector_field': {
                                'vector': embedding,
                                'k': k_pdfs * 2  # Get more candidates for diversity
                            }
                        }
                    })
                    
                    # Execute hybrid search
                    hybrid_results = client.search(
                        index=OPENSEARCH_INDEX,
                        body={
                            'query': {
                                'bool': {
                                    'should': search_queries,
                                    'minimum_should_match': 1
                                }
                            },
                            'size': k_pdfs * 3,  # Get more results for diversity
                            'collapse': {
                                'field': 'metadata.source.keyword'  # Collapse by source to ensure diversity
                            }
                        }
                    )
                    
                    # Process results and ensure diversity
                    seen_sources = set()
                    source_counts = {}
                    
                    for hit in hybrid_results['hits']['hits']:
                        source = hit['_source']
                        source_name = source['metadata']['source']
                        
                        # Count how many chunks we have from this source
                        current_count = source_counts.get(source_name, 0)
                        
                        # Limit to 3 chunks per source to ensure diversity
                        if current_count < 3:
                            chunk = {
                                'content': source['text'],
                                'metadata': source['metadata'],
                                'source_type': 'pdf_doc'
                            }
                            pdf_chunks.append(chunk)
                            source_counts[source_name] = current_count + 1
                            seen_sources.add(source_name)
                            
                            # Stop if we have enough total chunks
                            if len(pdf_chunks) >= k_pdfs:
                                break
                    
                    # If we still don't have enough results, try additional search strategies
                    if len(pdf_chunks) < k_pdfs // 2:
                        print("🔍 Trying additional search strategies...")
                        
                        # Strategy 1: Fall back to semantic search
                        try:
                            pdf_docs = self.pdf_vectorstore.similarity_search(enhanced_query, k=k_pdfs)
                            for doc in pdf_docs:
                                chunk = {
                                    'content': doc.page_content,
                                    'metadata': doc.metadata,
                                    'source_type': 'pdf_doc'
                                }
                                if not any(existing['content'] == chunk['content'] for existing in pdf_chunks):
                                    pdf_chunks.append(chunk)
                        except Exception as e:
                            print(f"⚠️ Semantic search fallback failed: {str(e)}")
                        
                        # Strategy 2: If we have potential file names but no results, try direct file search
                        if potential_file_names and len(pdf_chunks) < k_pdfs // 2:
                            print("🔍 Trying direct file name search...")
                            for file_name in potential_file_names:
                                try:
                                    file_results = client.search(
                                        index=OPENSEARCH_INDEX,
                                        body={
                                            'query': {
                                                'bool': {
                                                    'should': [
                                                        {'fuzzy': {'metadata.source': {'value': file_name, 'fuzziness': 'AUTO'}}},
                                                        {'match': {'metadata.source': file_name}},
                                                        {'wildcard': {'metadata.source': f'*{file_name}*'}}
                                                    ],
                                                    'minimum_should_match': 1
                                                }
                                            },
                                            'size': 5
                                        }
                                    )
                                    
                                    for hit in file_results['hits']['hits']:
                                        source = hit['_source']
                                        chunk = {
                                            'content': source['text'],
                                            'metadata': source['metadata'],
                                            'source_type': 'pdf_doc'
                                        }
                                        if not any(existing['content'] == chunk['content'] for existing in pdf_chunks):
                                            pdf_chunks.append(chunk)
                                            
                                        if len(pdf_chunks) >= k_pdfs:
                                            break
                                except Exception as e:
                                    print(f"⚠️ File name search failed for '{file_name}': {str(e)}")
                    
                    print(f"📄 Retrieved {len(pdf_chunks)} PDF/Doc chunks from {len(seen_sources)} different sources")
                    print(f"📋 Sources found: {list(seen_sources)}")
                    print(f"📊 Source counts: {source_counts}")
                except Exception as e:
                    print(f"❌ Error retrieving PDF chunks: {str(e)}")
                    # Fallback to simple semantic search
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
                    except Exception as fallback_error:
                        print(f"❌ Fallback search also failed: {str(fallback_error)}")
            
            # Retrieve from Sheets index
            if self.sheets_index:
                try:
                    # For sheet-specific queries, also try direct OpenSearch search
                    if is_sheet_query:
                        print("🔍 Using direct OpenSearch search for sheet data...")
                        client = create_opensearch_client()
                        
                        # Extract key terms from the query for targeted search
                        query_terms = enhanced_query.lower().split()
                        # Filter out common words and keep meaningful terms
                        meaningful_terms = [term for term in query_terms if len(term) > 3 and term not in ['the', 'and', 'for', 'with', 'this', 'that', 'file', 'containing', 'has', 'includes']]
                        
                        # Search for sheet content using hybrid approach
                        search_queries = [
                            {'match': {'content': enhanced_query}},
                            {'match_phrase': {'content': enhanced_query}}
                        ]
                        
                        # Add targeted searches for meaningful terms
                        for term in meaningful_terms:
                            search_queries.append({'match': {'content': term}})
                        
                        sheet_results = client.search(
                            index=SHEETS_INDEX,
                            body={
                                'query': {
                                    'bool': {
                                        'should': search_queries,
                                        'minimum_should_match': 1
                                    }
                                },
                                'size': k_sheets
                            }
                        )
                        
                        # Add direct search results
                        for hit in sheet_results['hits']['hits']:
                            source = hit['_source']
                            chunk = {
                                'content': source['content'],
                                'metadata': source.get('metadata', {}),
                                'source_type': 'sheet'
                            }
                            # Avoid duplicates
                            if not any(existing['content'] == chunk['content'] for existing in sheets_chunks):
                                sheets_chunks.append(chunk)
                    
                    # Also use LlamaIndex query engine for semantic search
                    query_engine = self.sheets_index.as_query_engine(
                        response_mode="tree_summarize",
                        similarity_top_k=k_sheets
                    )
                    sheets_response = query_engine.query(enhanced_query)
                    
                    # Extract source nodes (chunks) from response
                    if hasattr(sheets_response, 'source_nodes'):
                        for node in sheets_response.source_nodes:
                            chunk = {
                                'content': node.node.text,
                                'metadata': node.node.metadata,
                                'source_type': 'sheet'
                            }
                            # Avoid duplicates
                            if not any(existing['content'] == chunk['content'] for existing in sheets_chunks):
                                sheets_chunks.append(chunk)
                    
                    print(f"📊 Retrieved {len(sheets_chunks)} sheet chunks")
                except Exception as e:
                    print(f"❌ Error retrieving sheet chunks: {str(e)}")
            
            return pdf_chunks, sheets_chunks
            
        except Exception as e:
            print(f"❌ Error in unified retrieval: {str(e)}")
            return [], []

def create_mixed_query_response(pdf_chunks, sheets_chunks, query, llm, chat_history=None, user_email=None):
    """Create a response for mixed queries that combines content search with metadata formatting."""
    try:
        # Get file metadata for the found chunks
        enhanced_metadata = load_enhanced_file_metadata(user_email) if user_email else {}
        
        # Collect unique file sources from chunks
        file_sources = set()
        for chunk in pdf_chunks + sheets_chunks:
            if isinstance(chunk, dict):
                source = chunk.get('metadata', {}).get('source', 'Unknown')
                file_sources.add(source)
        
        # Get metadata for each file source
        file_metadata_list = []
        for source in file_sources:
            # Find metadata for this file
            for file_hash, metadata in enhanced_metadata.items():
                if metadata.get('file_name') == source:
                    file_metadata_list.append(metadata)
                    break
        
        # Create response with file metadata
        if file_metadata_list:
            response = f"**Files containing relevant content:**\n\n"
            for i, file_meta in enumerate(file_metadata_list, 1):
                file_name = file_meta.get('file_name', 'Unknown')
                file_type = file_meta.get('file_type', 'Unknown')
                web_link = file_meta.get('web_view_link', '')
                file_size = file_meta.get('file_size_mb', 0)
                
                response += f"{i}. **{file_name}** ({file_type.title()})\n"
                if web_link:
                    response += f"   🔗 [View in Google Drive]({web_link})\n"
                response += f"   📏 Size: {file_size:.2f} MB\n\n"
        else:
            response = "No files found containing the requested content."
        
        return response
        
    except Exception as e:
        print(f"❌ Error creating mixed query response: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"

def create_unified_response(pdf_chunks, sheets_chunks, query, llm, chat_history=None):
    """Create a unified response using all retrieved chunks in a single LLM call."""
    try:
        # Prepare context from PDF chunks
        pdf_context = ""
        if pdf_chunks:
            pdf_context = "**DOCUMENTS & PDFS:**\n"
            for i, chunk in enumerate(pdf_chunks, 1):
                if isinstance(chunk, dict):
                    content = chunk.get('content', '')
                    metadata = chunk.get('metadata', {})
                    source = metadata.get('source', 'Unknown')
                    page = metadata.get('page', 'N/A')
                    content_type = metadata.get('type', 'text')
                else:
                    content = str(chunk)
                    source = 'Unknown'
                    page = 'N/A'
                    content_type = 'text'
                pdf_context += f"\n--- Chunk {i} (Source: {source}, Page: {page}, Type: {content_type}) ---\n"
                pdf_context += content + "\n"
        
        # Prepare context from sheet chunks
        sheets_context = ""
        if sheets_chunks:
            sheets_context = "\n**SPREADSHEETS & EXCEL FILES:**\n"
            for i, chunk in enumerate(sheets_chunks, 1):
                if isinstance(chunk, dict):
                    content = chunk.get('content', '')
                    metadata = chunk.get('metadata', {})
                    source = metadata.get('source', 'Unknown')
                else:
                    content = str(chunk)
                    source = 'Unknown'
                sheets_context += f"\n--- Chunk {i} (Source: {source}) ---\n"
                sheets_context += content + "\n"
        
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

        # Get response from LLM using complete method for raw strings
        response = llm.complete(unified_prompt)
        return response.text
        
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

        # Scan for PDFs, Google Docs, and Sheets
        st.session_state.processing_status = "📁 Scanning your Google Drive for PDF, Google Doc, and Sheet files..."
        
        pdf_files = scan_entire_drive_for_pdfs(credentials)
        doc_files = scan_entire_drive_for_docs(credentials)
        sheet_files = scan_entire_drive_for_sheets(credentials)
        
        all_files = pdf_files + doc_files + sheet_files
        st.session_state.processing_status = f"📊 Found {len(pdf_files)} PDF files, {len(doc_files)} Google Doc files, and {len(sheet_files)} Sheet files in your Drive"
        
        # Clean up files that have been deleted from Google Drive
        deleted_files_cleaned = cleanup_deleted_files_from_metadata(all_files, user_email)
        
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
        modified_count = 0
        
        print(f"🔍 Processing {len(pdf_files)} PDF files, {len(doc_files)} Google Doc files, and {len(sheet_files)} Sheet files")
        
        # Process PDFs
        for pdf_file in pdf_files:
            file_id = pdf_file['id']
            file_name = pdf_file['name']
            
            # Check if file has been modified
            is_modified, existing_metadata = check_file_modification(pdf_file, user_email)
            
            if is_modified:
                print(f"🔄 File '{file_name}' has been modified, cleaning up old data...")
                cleanup_modified_file(file_name, user_email, file_id)
                modified_count += 1
            elif existing_metadata:
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
                    
                    # Extract text and images with enhanced metadata
                    pdf_docs = process_pdf_with_images(pdf_path, file_name, file_id, pdf_file)
                    if pdf_docs:
                        documents.extend(pdf_docs)
                        processed_count += 1
                        print(f"✅ Successfully processed {file_name}: {len(pdf_docs)} documents")
                    else:
                        failed_count += 1
                        
            except Exception as e:
                print(f"Error processing PDF {file_name}: {str(e)}")
                failed_count += 1
        
        # Process Google Docs
        for doc_file in doc_files:
            file_id = doc_file['id']
            file_name = doc_file['name']
            
            # Check if file has been modified
            is_modified, existing_metadata = check_file_modification(doc_file, user_email)
            
            if is_modified:
                print(f"🔄 File '{file_name}' has been modified, cleaning up old data...")
                cleanup_modified_file(file_name, user_email, file_id)
                modified_count += 1
            elif existing_metadata:
                print(f"⏭️ Skipping already processed Google Doc: {file_name}")
                skipped_count += 1
                continue
            
            try:
                st.session_state.processing_status = f"📝 Processing Google Doc: {file_name}"
                
                # Download Google Doc content with enhanced metadata
                doc = download_google_doc_content(service, file_id, file_name, doc_file)
                if doc:
                    documents.append(doc)
                    processed_count += 1
                    print(f"✅ Successfully processed {file_name}: 1 document")
                    # Process revisions (metadata only)
                    process_file_revisions(service, doc_file, user_email)
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"Error processing Google Doc {file_name}: {str(e)}")
                failed_count += 1
        
        # Process Sheets
        for sheet_file in sheet_files:
            file_id = sheet_file['id']
            file_name = sheet_file['name']
            
            # Check if file has been modified
            is_modified, existing_metadata = check_file_modification(sheet_file, user_email)
            
            if is_modified:
                print(f"🔄 File '{file_name}' has been modified, cleaning up old data...")
                cleanup_modified_file(file_name, user_email, file_id)
                modified_count += 1
            elif existing_metadata:
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
                                    # Add proper metadata to each sheet document
                                    for doc in sheet_documents:
                                        doc.metadata.update({
                                            'source': file_name,
                                            'file_id': file_id,
                                            'file_type': 'sheet',
                                            'processed_time': datetime.now().isoformat()
                                        })
                                    
                                    all_sheet_documents.extend(sheet_documents)
                                    processed_count += 1
                                    print(f"✅ Successfully processed {file_name}: {len(sheet_documents)} documents")
                                    
                                    # Extract enhanced metadata for sheets
                                    additional_info = {
                                        "sheet_count": 1,  # Default, could be enhanced with actual sheet count
                                        "row_count": 0,  # Could be enhanced with actual row count
                                        "column_count": 0,  # Could be enhanced with actual column count
                                        "text_chunks": len(sheet_documents),
                                        "total_chunks": len(sheet_documents)
                                    }
                                    
                                    # Store enhanced metadata
                                    file_hash = hashlib.md5(f"{file_id}_{sheet_file.get('modifiedTime', '')}".encode()).hexdigest()
                                    enhanced_metadata = extract_file_metadata(sheet_file, 'sheet', additional_info)
                                    save_enhanced_file_metadata({file_hash: enhanced_metadata}, user_email)
                                    
                                    # Process revisions (metadata only)
                                    process_file_revisions(service, sheet_file, user_email)
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
            success_message = f"✅ Successfully processed {processed_count} files ({failed_count} failed, {modified_count} modified)"
            if deleted_files_cleaned > 0:
                success_message += f", cleaned up {deleted_files_cleaned} deleted files"
            st.success(success_message)
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
                # Initialize LLM using LlamaIndex
                llm = initialize_llamaindex_llm()
                if not llm:
                    raise Exception("Failed to initialize LlamaIndex LLM")
                
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
        
        # Update session state
        status_message = "✅ Processing complete! You can now ask questions about your documents and sheets."
        if deleted_files_cleaned > 0:
            status_message += f" (Cleaned up {deleted_files_cleaned} deleted files)"
        st.session_state.processing_status = status_message
        
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
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "text"},
                                "file_id": {"type": "text"},
                                "file_type": {"type": "text"},
                                "processed_time": {"type": "text"}
                            }
                        },
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
        llm = initialize_llamaindex_llm()
        if not llm:
            raise Exception("Failed to initialize LlamaIndex LLM")
        
        # Create unified retriever
        print("🔗 Creating unified retriever...")
        unified_retriever = UnifiedRetriever(pdf_vectorstore, sheets_index, embeddings)
        
        # Update session state
        print("💾 Updating session state...")
        st.session_state.unified_retriever = unified_retriever
        st.session_state.llm = llm
        st.session_state.start_chatting = True
        
        print("✅ Unified retriever initialization completed successfully")
        return unified_retriever
        
    except Exception as e:
        print(f"❌ Error initializing unified retriever: {str(e)}")
        st.error(f"Error initializing unified retriever: {str(e)}")
        return False

def query_unified_system(prompt, unified_retriever, llm, chat_history=None, user_email=None):
    """Query the unified system using the new approach with query classification."""
    try:
        print("🔍 Querying unified system...")
        
        # Classify the query
        print("🎯 Classifying query...")
        classification = classify_query(prompt)
        query_type = classification.get('classification', 'content')
        confidence = classification.get('confidence', 0.0)
        reasoning = classification.get('reasoning', 'No reasoning provided')
        
        print(f"📊 Query classified as: {query_type} (confidence: {confidence:.2f})")
        print(f"💭 Reasoning: {reasoning}")
        
        # Route query based on classification
        if query_type == 'metadata':
            print("📋 Handling metadata query...")
            if user_email:
                response = handle_metadata_query(prompt, user_email)
            else:
                response = "❌ User email not available for metadata query processing."
        else:
            print("📄 Handling content query...")
            # Retrieve relevant chunks from both sources with history context
            pdf_chunks, sheets_chunks = unified_retriever.retrieve_relevant_chunks(prompt, k_pdfs=20, k_sheets=10, chat_history=chat_history)
            
            print(f"📄 Retrieved {len(pdf_chunks)} PDF chunks and {len(sheets_chunks)} sheet chunks")
            
            # Check if this is a mixed query (content search + metadata response)
            mixed_query_indicators = ["which file", "what file", "find the file", "show me the file", "which document", "what document"]
            is_mixed_query = any(indicator in prompt.lower() for indicator in mixed_query_indicators)
            
            if is_mixed_query and user_email:
                print("🔄 Detected mixed query - combining content search with metadata response...")
                # Create content-based response but format it to show file metadata
                response = create_mixed_query_response(pdf_chunks, sheets_chunks, prompt, llm, chat_history, user_email)
            else:
                # Create unified response in a single LLM call with history
                response = create_unified_response(pdf_chunks, sheets_chunks, prompt, llm, chat_history)
        
        print("✅ Unified response generated successfully")
        return response
        
    except Exception as e:
        print(f"❌ Error in unified query system: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"

def process_file_revisions(service, file_info, user_email):
    """
    Fetch and process revisions for a Google Doc or Sheet file.
    - Stores revision metadata as part of the file's comprehensive metadata
    - Does NOT create separate content chunks for revisions
    """
    file_id = file_info['id']
    file_name = file_info['name']
    mime_type = file_info.get('mimeType', '')
    file_type = 'google_doc' if mime_type == 'application/vnd.google-apps.document' else (
        'sheet' if mime_type == 'application/vnd.google-apps.spreadsheet' else None)
    if not file_type:
        return  # Only process Google Docs/Sheets

    try:
        revisions = service.revisions().list(fileId=file_id, fields="revisions(id, modifiedTime, size, keepForever, originalFilename, mimeType, lastModifyingUser)").execute().get('revisions', [])
        # Only process the 5 most recent revisions for efficiency
        revisions = sorted(revisions, key=lambda r: r.get('modifiedTime', ''), reverse=True)
        
        revision_list = []
        for rev in revisions:
            rev_info = {
                'revision_id': rev.get('id'),
                'revision_modified_time': rev.get('modifiedTime', ''),
                'revision_size': rev.get('size', ''),
                'revision_keep_forever': rev.get('keepForever', False),
                'revision_original_filename': rev.get('originalFilename', ''),
                'revision_mime_type': rev.get('mimeType', ''),
                'last_modifying_user': rev.get('lastModifyingUser', {})
            }
            revision_list.append(rev_info)
        
        # Update the file's metadata with revision information
        file_hash = hashlib.md5(f"{file_id}_{file_info.get('modifiedTime', '')}".encode()).hexdigest()
        
        # Load existing metadata
        enhanced_metadata = load_enhanced_file_metadata(user_email)
        if file_hash in enhanced_metadata:
            # Update existing metadata with revision info
            enhanced_metadata[file_hash]['revisions'] = revision_list
            enhanced_metadata[file_hash]['revision_count'] = len(revision_list)
        else:
            # Create new metadata entry with revision info
            file_metadata = extract_file_metadata(file_info, file_type)
            file_metadata['revisions'] = revision_list
            file_metadata['revision_count'] = len(revision_list)
            enhanced_metadata[file_hash] = file_metadata
        
        # Save updated metadata
        save_enhanced_file_metadata(enhanced_metadata, user_email)
        
        print(f"✅ Added {len(revision_list)} revisions to metadata for {file_name}")
        
    except Exception as e:
        print(f"Error processing revisions for {file_name}: {str(e)}")

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
                # Clear enhanced metadata
                enhanced_metadata_path = get_user_enhanced_metadata_path(user_email)
                if os.path.exists(enhanced_metadata_path):
                    os.remove(enhanced_metadata_path)
                
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
        
        # Show enhanced metadata information
        enhanced_metadata = load_enhanced_file_metadata(user_email)
        if enhanced_metadata:
            with st.expander("📋 File Metadata Summary", expanded=False):
                total_files = len(enhanced_metadata)
                file_types = {}
                total_size_mb = 0
                total_chunks = 0
                
                for file_hash, file_data in enhanced_metadata.items():
                    file_type = file_data.get('file_type', 'unknown')
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    total_size_mb += file_data.get('file_size_mb', 0)
                    total_chunks += file_data.get('total_chunks', 0)
                
                st.write(f"**Total Files:** {total_files}")
                st.write(f"**Total Size:** {total_size_mb:.2f} MB")
                st.write(f"**Total Chunks:** {total_chunks}")
                
                st.write("**File Types:**")
                for file_type, count in file_types.items():
                    st.write(f"  • {file_type.title()}: {count}")
                
                # Show recent files
                recent_files = sorted(
                    enhanced_metadata.items(),
                    key=lambda x: x[1].get('processed_time', ''),
                    reverse=True
                )[:5]
                
                st.write("**Recently Processed:**")
                for file_hash, file_data in recent_files:
                    st.write(f"  • {file_data.get('file_name', 'Unknown')} ({file_data.get('file_type', 'unknown')})")
        
        # Debug information
        with st.expander("🔧 Debug Info"):
            st.write(f"processed_pdfs: {st.session_state.processed_pdfs}")
            st.write(f"processed_sheets: {st.session_state.processed_sheets}")
            st.write(f"start_chatting: {st.session_state.start_chatting}")
            st.write(f"has_unified_retriever: {st.session_state.unified_retriever is not None}")
            st.write(f"has_llm: {st.session_state.llm is not None}")
            st.write(f"pdf_chunks: {pdf_count}")
            st.write(f"sheet_chunks: {sheets_count}")
            st.write(f"enhanced_metadata_files: {len(enhanced_metadata) if enhanced_metadata else 0}")
        
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
                        # Classify the query first for debugging
                        classification = classify_query(prompt)
                        query_type = classification.get('classification', 'content')
                        confidence = classification.get('confidence', 0.0)
                        
                        # Show classification info in debug mode
                        with st.expander("🔍 Query Classification Info", expanded=False):
                            st.write(f"**Query Type:** {query_type}")
                            st.write(f"**Confidence:** {confidence:.2f}")
                            st.write(f"**Reasoning:** {classification.get('reasoning', 'No reasoning')}")
                        
                        # Use unified query system with user email
                        answer = query_unified_system(
                            prompt, 
                            st.session_state.unified_retriever, 
                            st.session_state.llm,
                            st.session_state.messages,
                            user_email
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

def cleanup_deleted_files_from_metadata(drive_files, user_email):
    """
    Clean up files that exist in metadata but are no longer present in Google Drive.
    
    Args:
        drive_files (list): List of file info dictionaries from Google Drive
        user_email (str): User's email
        
    Returns:
        int: Number of files cleaned up
    """
    try:
        print("🔍 Checking for files that have been deleted from Google Drive...")
        
        # Load enhanced metadata
        enhanced_metadata = load_enhanced_file_metadata(user_email)
        if not enhanced_metadata:
            print("✅ No metadata found, nothing to clean up")
            return 0
        
        # Get all file IDs from Google Drive
        drive_file_ids = {file['id'] for file in drive_files}
        
        # Find files in metadata that are no longer in Drive
        files_to_cleanup = []
        for file_hash, metadata in enhanced_metadata.items():
            file_id = metadata.get('file_id')
            file_name = metadata.get('file_name')
            
            if file_id and file_id not in drive_file_ids:
                files_to_cleanup.append({
                    'file_id': file_id,
                    'file_name': file_name,
                    'file_hash': file_hash
                })
        
        if not files_to_cleanup:
            print("✅ All files in metadata are still present in Google Drive")
            return 0
        
        print(f"🗑️ Found {len(files_to_cleanup)} files that have been deleted from Google Drive")
        
        # Clean up each deleted file
        cleaned_count = 0
        for file_info in files_to_cleanup:
            file_id = file_info['file_id']
            file_name = file_info['file_name']
            file_hash = file_info['file_hash']
            
            print(f"🧹 Cleaning up deleted file: {file_name} (ID: {file_id})")
            
            # Delete chunks from PDF/Docs index
            pdf_cleanup_success = delete_chunks_from_opensearch_by_filename(file_name, OPENSEARCH_INDEX)
            
            # Delete chunks from sheets index
            sheets_cleanup_success = delete_chunks_from_opensearch_by_filename(file_name, SHEETS_INDEX)
            
            # Delete from enhanced metadata JSON file
            enhanced_metadata_cleanup_success = delete_file_from_enhanced_metadata(file_name, user_email)
            
            # Delete from OpenSearch metadata index
            metadata_index_cleanup_success = delete_metadata_from_opensearch_by_file_id(file_id, user_email)
            
            if pdf_cleanup_success or sheets_cleanup_success or enhanced_metadata_cleanup_success or metadata_index_cleanup_success:
                print(f"✅ Cleanup completed for deleted file: {file_name}")
                cleaned_count += 1
            else:
                print(f"⚠️ No cleanup actions were performed for deleted file: {file_name}")
        
        print(f"✅ Cleaned up {cleaned_count} deleted files from all indexes and metadata")
        return cleaned_count
        
    except Exception as e:
        print(f"❌ Error cleaning up deleted files: {str(e)}")
        return 0

if __name__ == "__main__":
    main() 
