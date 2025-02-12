import os
import pickle
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Set up Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Load FAISS vector store
faiss_index_path = "./faiss_index/faiss_store.pkl"
with open(faiss_index_path, "rb") as f:
    vectorstore = pickle.load(f)

# Create retriever with optimized settings
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Fetch more chunks for better answers
)

# Set up Gemini and chain
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
prompt = PromptTemplate.from_template("""You are an AI assistant. Answer the question as best as you can with the provided context and conversation history.

Conversation History:
{history}

Context: {context}
Question: {input}

Answer: """)

# -prompt = PromptTemplate.from_template("""You are an AI assistant. Answer the question as best as you can.
# -If the provided context and conversation history don't contain enough information to answer the question, 
# -provide a helpful response based on your general knowledge while mentioning that you're answering based on general knowledge rather than specific context.

# Create chains
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

class ConversationalQA:
    def __init__(self):
        self.history = []  # Stores conversation history

    def ask_question(self, query):
        # Keep last 5 exchanges for context
        history = "\n".join(self.history[-5:]) if self.history else "No previous conversation."
        
        response = retrieval_chain.invoke({
            "history": history, 
            "context": history,  # Merge history into context explicitly
            "input": query
        })

        # Store question and answer in history
        self.history.append(f"Q: {query}\nA: {response['answer']}")
        
        return response["answer"]

# Interactive loop with memory
if __name__ == "__main__":
    chat = ConversationalQA()
    print("Welcome to the Q&A system! Type 'quit' to exit.")

    while True:
        question = input("\nWhat's your question? ")
        if question.lower() == 'quit':
            break
        answer = chat.ask_question(question)
        print("\nAnswer:", answer)
