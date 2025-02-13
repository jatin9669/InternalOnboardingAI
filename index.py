import os
import pickle
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Define paths
KB_FOLDER = "./KB"
FAISS_INDEX_PATH = "./faiss_index/faiss_store.pkl"

# Function to process and index PDFs
def process_pdfs():
    documents = []
    for file in os.listdir(KB_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(KB_FOLDER, file)
            loader = PyMuPDFLoader(pdf_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save FAISS index
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(vectorstore, f)
    
    print("PDFs successfully processed and indexed.")

if __name__ == "__main__":
    process_pdfs()
