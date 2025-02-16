import os
import pickle
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
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
    search_kwargs={"k": 5}  # Fetch more chunks for better answers
)

# Set up Gemini and chain
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# template
template_for_answer = """Answer the following question based on the provided context with proper citations.
<context>
{context}
</context>

Question:{input}
"""
prompt_for_history = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# Create chains
document_chain = create_stuff_documents_chain(llm, answer_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
history_retriever_chain = create_history_aware_retriever(llm,retriever,prompt_for_history)

conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)

class ConversationalQA:
    def __init__(self):
        self.history = []  # Stores conversation history

    def ask_question(self, query):
        response = conversational_retrieval_chain.invoke({'chat_history': self.history, 'input': query})
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
