import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define persist directory
persist_directory = "./chroma_db"

# Delete existing vectorstore if it exists
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# Load and process data from all PDF files in data_files directory
loader = DirectoryLoader(
    './data_files', 
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# Print number of PDFs loaded
print(f"\nLoaded {len(documents)} PDF documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Print number of chunks created
print(f"Split into {len(docs)} chunks")

# Generate embeddings and create vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
vectorstore.persist()
