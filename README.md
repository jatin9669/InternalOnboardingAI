# Personal AI Assistant with Google Drive Integration

A comprehensive AI-powered assistant that uses Google SSO to access your entire Google Drive, processes all your PDF documents, and creates a personalized knowledge base for intelligent Q&A.

## Features

- 🔐 **Google SSO Authentication**: Secure sign-in with your Google account
- 🔍 **Complete Drive Scanning**: Automatically finds and processes ALL PDFs in your Drive
- 🧠 **Personal Knowledge Base**: Creates embeddings from your entire document collection
- 💬 **Intelligent Chat**: Ask questions about any of your documents
- 🔒 **User-Specific Data**: Each user gets their own private vector database
- 📊 **Progress Tracking**: Real-time progress during document processing
- 🔄 **Auto-Refresh**: Option to update your knowledge base with new documents

## Running the project
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Environment Setup

### 1. Create .env file:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Google OAuth Configuration
GOOGLE_CLIENT_ID=your_google_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

### 2. Set up Google OAuth:

1. **Go to [Google Cloud Console](https://console.cloud.google.com/)**
2. **Create a new project** or select existing one
3. **Enable APIs**:
   - Google Drive API
   - Google+ API (for user info)
4. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth 2.0 Client IDs"
   - Application type: "Web application"
   - Add authorized redirect URI: `http://localhost:8501`
5. **Copy Client ID and Secret** to your `.env` file

## How to Use

### Step 1: Sign in with Google
1. Run the Streamlit app
2. Click "Sign in with Google"
3. Complete the OAuth flow
4. Grant permission to read your Google Drive

### Step 2: Process Your Documents
1. Click "Process My Google Drive"
2. Wait while the app scans your entire Drive for PDFs
3. Watch the progress as documents are processed
4. Your personal knowledge base is built automatically

### Step 3: Chat with Your Documents
1. Start asking questions about any of your documents
2. The AI will search across your entire document collection
3. Get intelligent answers with source references

## Technical Stack

- **Frontend**: Streamlit
- **Authentication**: Google OAuth 2.0
- **PDF Processing**: PyMuPDF (fitz)
- **Vector Search**: FAISS (user-specific databases)
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Drive Integration**: Google Drive API

## Key Features

### 🔐 Secure Authentication
- Google OAuth 2.0 integration
- No need to make files public
- Secure access to user's private Drive

### 🧠 Intelligent Processing
- Scans entire Google Drive automatically
- Processes all PDF files in user's account
- Creates user-specific vector databases
- Preserves document metadata and links

### 💾 Personal Data Management
- Each user gets their own vector database
- Persistent storage across sessions
- Option to refresh with new documents
- Complete privacy and data isolation

### 💬 Advanced Chat Features
- Context-aware conversations
- Source document references
- Intelligent retrieval across all user documents
- Memory of chat history

