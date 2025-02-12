import pickle
from langchain_community.vectorstores import FAISS

# Load the FAISS index
faiss_index_path = "./faiss_index/faiss_store.pkl"
with open(faiss_index_path, "rb") as f:
    vectorstore = pickle.load(f)
    # Fetch and display some stored chunks


# Check the number of stored embeddings
print(f"Total vectors stored in FAISS: {vectorstore.index.ntotal}")

docs = vectorstore.similarity_search("test", k=3)  # Test query to fetch 3 chunks
for i, doc in enumerate(docs):
    print(f"\n🔹 Chunk {i+1}:")
    print(doc.page_content)