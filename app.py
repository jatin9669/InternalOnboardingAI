import streamlit as st
import os
import re
import fitz  # PyMuPDF for PDF processing
import pdfplumber  # For table extraction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from logger import LoggingCallbacks
from excel import write_to_excel
import tiktoken
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize session state
st.session_state.setdefault('processed_pdfs', False)
st.session_state.setdefault('chat_history', [])
st.session_state.setdefault('qa_chain', None)
st.session_state.setdefault('vectorstore', None)

def count_tokens(text, model="text-embedding-3-small"):
    """Counts tokens in a given text using OpenAI's tokenizer."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def calculate_cost(total_tokens):
    """Calculates the cost of the query based on the total tokens."""
    return (total_tokens / 1000) * 0.00002

def extract_text_and_tables(pdf_path):
    """Extract text, hyperlinks, and tables from a PDF."""
    doc = fitz.open(pdf_path)
    extracted_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            links = [link["uri"] for link in page.get_links() if "uri" in link]
            text_links = list(set(re.findall(r"https?://\S+", page_text)))
            links.extend(text_links)
            tables = pdf.pages[page_num].extract_table() if page_num < len(pdf.pages) else None
            extracted_data.append({
                "page": page_num + 1,
                "text": page_text,
                "links": links,
                "tables": tables
            })
    return extracted_data

def process_pdfs(uploaded_files):
    """Process PDFs, extract text, tables & links, and store in FAISS."""
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            extracted_pages = extract_text_and_tables(temp_path)
            from langchain_core.documents import Document
            for page in extracted_pages:
                combined_text = page["text"] + "\n\n🔗 Links:\n" + "\n".join(page["links"])
                if page["tables"]:
                    combined_text += "\n\n📊 Table:\n" + str(page["tables"])
                documents.append(Document(page_content=combined_text))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=500, separators=["\n\n", "\n", " ", ""], keep_separator=True
    )
    splits = text_splitter.split_documents(documents)
    
    total_tokens = 0
    for split in splits:
        total_tokens += count_tokens(split.page_content) # for processing pdfs
    
    estimated_cost = calculate_cost(total_tokens)
    write_to_excel(total_tokens, "N/A", total_tokens, estimated_cost, "Embeddings")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(splits, embeddings)
    st.session_state.vectorstore = vectorstore  # Store FAISS globally
    return vectorstore

def setup_qa_chain(vectorstore):
    """Set up the QA chain with FAISS retriever."""
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, callbacks=[LoggingCallbacks()])
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    prompt_for_history = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Generate a search query based on the conversation history and question.")
    ])
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful technical assistant. Provide clear, accurate answers based on the documentation.
        - If the context contains URLs, **always include them**.
        - Format URLs as proper clickable links.
        - If referring to a table, summarize key insights.
        Context: {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_for_history)
    return create_retrieval_chain(history_aware_retriever, document_chain)

def main():
    st.set_page_config(page_title="Team Documentation Assistant", page_icon="📚", layout="wide")
    st.title("📚 Team Documentation Assistant")

    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader("Upload PDF Documents", type=['pdf'], accept_multiple_files=True)
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                vectorstore = process_pdfs(uploaded_files)
                st.session_state.qa_chain = setup_qa_chain(vectorstore)
                st.session_state.processed_pdfs = True
                st.success(f"Processed {len(uploaded_files)} documents successfully!")
        if st.session_state.processed_pdfs and st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    if not st.session_state.processed_pdfs:
        st.info("👆 Please upload PDF documents and click 'Process Documents' to start.")
        return

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        retriever = st.session_state.vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                formatted_history = [(msg["role"], msg["content"]) for msg in st.session_state.chat_history[:-1]]
                response = st.session_state.qa_chain.invoke({"chat_history": formatted_history, "input": prompt})
                with st.expander("Retrieved Chunks"):
                    for doc in retrieved_docs:
                        st.write(doc.page_content)
                st.write(response["answer"])
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()
