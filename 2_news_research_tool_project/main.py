import streamlit as st
import os
import pickle
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
os.environ['OPENAI_API_KEY'] = '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'
load_dotenv()

# Set up Streamlit UI
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs from the user
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data using SeleniumURLLoader
    loader = SeleniumURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings using OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    main_placeholder.text("Embedding Vector Building...Started...✅✅✅")

    # Create FAISS index from embeddings and save it
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    vectorstore_openai.save_local("faiss_store")
    main_placeholder.text("Embedding Vector Saved...✅✅✅")
    time.sleep(2)

query = main_placeholder.text_input("Question: ")
if query:
    # Ensure embeddings is properly assigned
    embeddings = OpenAIEmbeddings()

    # Load FAISS index with dangerous deserialization allowed
    vectorIndex = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)

    # Create chain with loaded index
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)

    # Display answer
    st.header("Answer")
    st.write(result["answer"])

    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")
        for source in sources_list:
            st.write(source)
