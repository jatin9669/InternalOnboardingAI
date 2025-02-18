import streamlit as st
import os
import pickle
import fitz  # PyMuPDF for PDF processing
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import re

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

def extract_text_with_links(pdf_path):
    """Extract text and hyperlinks from a PDF."""
    doc = fitz.open(pdf_path)
    extracted_data = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        links = []

        # Extract clickable links
        for link in page.get_links():
            if "uri" in link:
                links.append(link["uri"])
                page_text += f"\n[🔗 Link: {link['uri']}]"

        # Extract plain text URLs using regex
        url_pattern = r"https?://\S+"  
        text_links = re.findall(url_pattern, page_text)
        links.extend(text_links)

        # Remove duplicates
        links = list(set(links))

        extracted_data.append({
            "page": page_num + 1,
            "text": page_text,
            "links": links
        })

    return extracted_data

def process_pdfs(uploaded_files):
    """Process PDFs, extract text & links, and store in FAISS."""
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())

            # Extract text with links
            extracted_pages = extract_text_with_links(temp_path)

            # Debug: Print extracted text and links
            for page in extracted_pages:
                print(f"\n📄 Page {page['page']} Text (First 500 chars):\n{page['text'][:500]}")
                print(f"🔗 Extracted Links: {page['links']}")

            # Ensure links stay with the text
            from langchain_core.documents import Document
            for page in extracted_pages:
                combined_text = page["text"] + "\n\n🔗 Links:\n" + "\n".join(page["links"])
                documents.append(Document(page_content=combined_text))

    # Split documents (ensure links stay together)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    splits = text_splitter.split_documents(documents)

    print(f"\n=== Number of chunks created: {len(splits)} ===\n")

    # Create embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(splits, embeddings)
    st.session_state.vectorstore = vectorstore  # Store FAISS globally
    return vectorstore


def setup_qa_chain(vectorstore):
    """Set up the QA chain with FAISS retriever."""
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

    # Set up retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Define prompt for history-aware retrieval
    prompt_for_history = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Generate a search query based on the conversation history and question.")
    ])

    # Define answer prompt (force LLM to include links)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful technical assistant. Provide clear, accurate answers based on the documentation.
        - If the context contains URLs, **always include them**.
        - Format URLs as proper clickable links.
        - If referring to a specific section with a link, mention both the section name and the link.

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
        page_title="Teamef Documentation Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Teamef Documentation Assistant")

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

        # Retrieve documents from FAISS
        retriever = st.session_state.vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(prompt)

        # Debug retrieved chunks
        print("\n=== Retrieved Chunks ===")
        for i, doc in enumerate(retrieved_docs[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:\n", doc.page_content[:2000])  # First 2000 chars

        # Generate response from LLM
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                formatted_history = [(msg["role"], msg["content"]) for msg in st.session_state.chat_history[:-1]]
                response = st.session_state.qa_chain.invoke({
                    "chat_history": formatted_history,
                    "input": prompt
                })

                print("\n=== LLM Final Answer ===\n", response["answer"])
                st.write(response["answer"])

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()
