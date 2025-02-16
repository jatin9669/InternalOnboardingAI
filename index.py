import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
import pickle
# Define FAISS index path
faiss_index_path = "./faiss_index"

# Delete existing FAISS index if it exists
# if os.path.exists(faiss_index_path):
#     shutil.rmtree(faiss_index_path)

# Load and process PDF
pdf_path = "./KB/dynamo.pdf"  
loader = DirectoryLoader(
    './data_files', 
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader
)
documents = loader.load()

# Efficiently split text for retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Large chunks retain better context
    chunk_overlap=400
)
docs = text_splitter.split_documents(documents)

# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Save FAISS index using pickle
with open(f"{faiss_index_path}/faiss_store.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

print("PDF successfully processed and stored in FAISS vector database.")
