import os
import pickle
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Load FAISS vector store
faiss_index_path = "./faiss_index/faiss_store.pkl"
with open(faiss_index_path, "rb") as f:
    vectorstore = pickle.load(f)

# Setup retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieves top 5 similar chunks
)

# Initialize OpenAI GPT model
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Define Prompt Template
prompt = PromptTemplate.from_template("""
You are an AI assistant. Answer the question using only the provided context and conversation history.
If the answer is not in the context, respond with "I don't know."

Conversation History:
{history}

Context:
{context}

Question: {input}

Answer:
""")

# Create Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    condense_question_llm=llm,  # Rephrases queries for better retrieval
)

class ConversationalQA:
    def __init__(self):
        self.history = []  # Stores conversation history

    def ask_question(self, query):
        # Convert last 5 exchanges into tuple format [(user, assistant), ...]
        history_tuples = [
            (q.split("\nA: ")[0].replace("Q: ", ""), q.split("\nA: ")[1])
            for q in self.history[-5:]
            if "\nA: " in q
        ]

        response = qa_chain.invoke({
            "chat_history": history_tuples,
            "question": query
        })

        # Store the conversation
        self.history.append(f"Q: {query}\nA: {response['answer']}")
        
        return response["answer"]

# Interactive chat loop
if __name__ == "__main__":
    chat = ConversationalQA()
    print("Welcome to the Q&A system! Type 'quit' to exit.")

    while True:
        question = input("\nWhat's your question? ")
        if question.lower() == 'quit':
            break
        answer = chat.ask_question(question)
        print("\nAnswer:", answer)
