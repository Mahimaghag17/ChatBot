import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings   # <-- NEW


# Streamlit UI
st.header("My first Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")


# Extract text from PDF
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""   # safer than assuming text exists

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # ---- HuggingFace embeddings ----
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding)

    # User question input
    user_question = st.text_input("Type your question here")

    if user_question:
        # Similarity search in FAISS
        match = vector_store.similarity_search(user_question)

        # Ollama LLM (running locally)
        llm = Ollama(model="llama2")   # or mistral, codellama, etc.

        # QA chain (stuff: simplest type)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)

        st.write(response)
