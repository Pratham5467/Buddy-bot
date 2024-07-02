import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

st.title("BUDDY BOT")

# File uploader for PDFs
uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    if "vector" not in st.session_state:
        try:
             st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Failed to create embeddings. Error: {str(e)}")
            st.error("Please check your internet connection and ensure the required packages are installed.")
            st.stop()
            
        
        # Load and process PDFs
        documents = []
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(uploaded_file.name)
            documents.extend(loader.load())
            
            # Remove temporary file
            os.remove(uploaded_file.name)

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    llm = ChatGroq(groq_api_key=groq_api_key,
                   model_name="mixtral-8x7b-32768")

    prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_prompt = st.text_input("Input your prompt here")

    if user_prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
else:
    st.write("Please upload PDF files to proceed.")