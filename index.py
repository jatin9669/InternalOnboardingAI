import os
import hashlib
from langchain_community.document_loaders import TextLoader
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

# Load and process data
file_path = "./data.txt"
new_hash = compute_file_hash(file_path)

# Check if file has changed
if os.path.exists(hash_file):
    with open(hash_file, "r") as f:
        old_hash = f.read().strip()
else:
    old_hash = ""

# Initialize ChromaDB
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
