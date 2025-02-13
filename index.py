import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define persist directory
persist_directory = "./chroma_db"
hash_file = "./data_hash.txt"

# Function to compute file hash
def compute_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

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
vectorstore = Chroma(persist_directory=persist_directory, 
                    embedding_function=embeddings,
                    collection_name="my_collection")

if new_hash != old_hash:
    print(f"File '{file_path}' has changed. Updating ChromaDB...")
    
    # Load and process data
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Clear and update the collection
    vectorstore.delete_collection()
    vectorstore = Chroma(persist_directory=persist_directory, 
                        embedding_function=embeddings,
                        collection_name="my_collection")
    vectorstore.add_documents(docs)

    # Save new hash
    with open(hash_file, "w") as f:
        f.write(new_hash)
else:
    print(f"No changes detected in '{file_path}'. Skipping update.")
