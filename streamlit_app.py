import streamlit as st
import os
import pickle
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
        chunk_size=2000, # TODO: experiment with different values
        chunk_overlap=400,
        length_function=len, # used for splitting the documents into chunks based on the length of the documents (len here is the length of the document)
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], 
    )
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    """Set up the QA chain with the vectorstore"""

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        # temperature=0.1,
        api_key=OPENAI_API_KEY
    )
    
    # Set up retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            # "score_threshold": 0.75 
        }
    )

    # Prompt for search query based on the conversation history and question.
    prompt_for_search = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Generate a search query based on the conversation history and question.")
    ])

    # Prompt for answer based on the context and the question.
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful technical assistant. Provide clear, accurate answers based on the documentation context.
        Include specific references when possible. If uncertain, acknowledge it and stick to the provided information.

        Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    # Create chains
    document_chain = create_stuff_documents_chain(
        llm=llm, 
        prompt=answer_prompt,
        document_variable_name="context" # it is the variable name for the context in the answer_prompt used to inject the context into the prompt.
    )

    # Create history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, 
        retriever=retriever,
        prompt=prompt_for_search
    )
    
    return create_retrieval_chain(
        retriever=history_aware_retriever, # it is the retriever that will be used to retrieve the context.
        combine_docs_chain=document_chain # it is the chain that will be used to answer the question. It is the chain that will be used to combine the context and the question.
    )

def main():
    st.set_page_config(
        page_title="Team Documentation Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Team Documentation Assistant")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload Team Documentations",
            type=['pdf'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    st.session_state.vectorstore = process_pdfs(uploaded_files)
                    st.session_state.qa_chain = setup_qa_chain(st.session_state.vectorstore)
                    st.success(f"✅ Processed {len(uploaded_files)} documents successfully!")

        
        if st.session_state.vectorstore is not None:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

    # Main chat interface
    if st.session_state.vectorstore is None:
        st.info("👆 Please upload team documentation PDFs to begin.")
        return

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your dorcuments..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching for relevant information..."):
                formatted_history = [
                    (msg["role"], msg["content"]) 
                    for msg in st.session_state.chat_history[-6:] # last 6 messages (TODO: experiment with different values)
                ]
                
                try:
                    response = st.session_state.qa_chain.invoke({
                        "chat_history": formatted_history,
                        "input": prompt
                    })
                    st.expander("View Context Used").write(response["context"])
                    st.markdown(response["answer"])
                    
                    # Add source documents expander
                    if "source_documents" in response:
                        with st.expander("📑 Source References"):
                            for i, doc in enumerate(response["source_documents"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"]
                    })
                    
                except Exception as e:
                    st.error("I apologize, but I couldn't find relevant information in the documentation. Please try rephrasing your question or ask about a different topic.")

if __name__ == "__main__":
    main() 