import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = []


def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()

        # Get text from main content areas
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        return text.strip()
    except Exception as e:
        st.error(f"Error processing {url}: {str(e)}")
        return None


def process_urls(urls):
    all_texts = []
    for url in urls:
        if url not in st.session_state.processed_urls:
            with st.spinner(f"Processing {url}..."):
                text = extract_text_from_url(url)
                if text:
                    all_texts.append(text)
                    st.session_state.processed_urls.append(url)

    if all_texts:
        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.create_documents(all_texts)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()  # or HuggingFaceEmbeddings()
        st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

        st.success(f"Processed {len(urls)} URLs. You can now ask questions!")


def answer_question(question):
    if not st.session_state.vector_store:
        st.error("Please process some URLs first!")
        return

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=st.session_state.vector_store.as_retriever(),
        return_source_documents=True
    )

    with st.spinner("Searching for answers..."):
        result = qa({"query": question})
        answer = result["result"]
        sources = result["source_documents"]

    st.subheader("Answer")
    st.write(answer)

    if st.checkbox("Show sources"):
        st.subheader("Source Documents")
        for i, doc in enumerate(sources):
            st.write(f"**Source {i + 1}**")
            st.write(doc.page_content[:500] + "...")  # Show first 500 chars
            st.write("---")


# Streamlit UI
st.title("Web Content Q&A Tool")
st.markdown("Enter URLs and ask questions based on their content")

# URL input
urls = st.text_area(
    "Enter URLs (one per line)",
    height=150,
    help="Paste one or more URLs to analyze, separated by new lines"
)

if st.button("Process URLs"):
    url_list = [url.strip() for url in urls.split('\n') if url.strip()]
    if url_list:
        process_urls(url_list)
    else:
        st.error("Please enter at least one valid URL")

# Question input
question = st.text_input(
    "Ask a question about the content",
    placeholder="Type your question here...",
    disabled=not st.session_state.vector_store
)

if question:
    answer_question(question)

# Display processed URLs
if st.session_state.processed_urls:
    st.sidebar.subheader("Processed URLs")
    for url in st.session_state.processed_urls:
        st.sidebar.write(f"- {url}")