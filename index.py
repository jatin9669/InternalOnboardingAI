import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define persist directory
persist_directory = "./chroma_db"

# Delete existing vectorstore if it exists
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# Load and process data
loader = TextLoader("./data.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Generate embeddings and create vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
vectorstore.persist()
