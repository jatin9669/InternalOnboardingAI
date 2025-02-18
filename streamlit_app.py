import streamlit as st
import os
import pickle
import tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from logging_call_backs import LoggingCallbacks
from write_to_excel import write_to_excel

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize session state
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None


def count_tokens(text, model="text-embedding-3-small"):
    """Counts tokens in a given text using OpenAI's tokenizer."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def process_pdfs(uploaded_files):
    """Process uploaded PDFs and create FAISS index"""
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            loader = PyMuPDFLoader(temp_path)
            documents.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
    )
    splits = text_splitter.split_documents(documents)

    # Count tokens in all chunks
    total_tokens = 0
    for split in splits:
        total_tokens += count_tokens(split.page_content)

    # Estimate cost
    estimated_cost = (total_tokens / 1000) * 0.00002
    print(f"Total tokens while processing pdfs: {total_tokens}")
    print(f"Estimated cost for embeddings: ${estimated_cost}")

    # Write to excel
    write_to_excel("N/A", "N/A", total_tokens, estimated_cost, "Embeddings")

#     embeddings = HuggingFaceEmbeddings(
# -        model_name="sentence-transformers/all-MiniLM-L6-v2"
# -    )
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    """Set up the QA chain with the vectorstore"""

    #llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, callbacks=[LoggingCallbacks()])
    
    # Set up retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Create promptsNo
    prompt_for_history = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Generate a search query based on the conversation history and question.")
    ])

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful technical assistant. Provide clear, accurate answers based on the documentation context.
        Include specific references when possible. If uncertain, acknowledge it and stick to the provided information.
        
        Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # Create chains
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_for_history)
    
    return create_retrieval_chain(history_aware_retriever, document_chain)

def main():
    st.set_page_config(
        page_title="Team Documentation Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Team Documentation Assistant")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    vectorstore = process_pdfs(uploaded_files)
                    st.session_state.qa_chain = setup_qa_chain(vectorstore)
                    st.session_state.processed_pdfs = True
                    st.success(f"Processed {len(uploaded_files)} documents successfully!")

        if st.session_state.processed_pdfs:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

    # Main chat interface
    if not st.session_state.processed_pdfs:
        st.info("👆 Please upload PDF documents and click 'Process Documents' to start.")
        return

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                formatted_history = [
                    (msg["role"], msg["content"]) 
                    for msg in st.session_state.chat_history[:-1]
                ]
                # context
                response = st.session_state.qa_chain.invoke({
                    "chat_history": formatted_history,
                    "input": prompt
                })
                st.expander("View Context Used").write(response["context"])
                st.write(response["answer"])
                
        # Add assistant response to chat history
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response["answer"]}
        )

if __name__ == "__main__":
    main() 