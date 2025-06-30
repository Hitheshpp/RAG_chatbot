from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from mongo_data_fetching import get_documents
from gradio_mistral_llm import GradioMistralLLM
import hashlib
import streamlit as st

# Function to hash MongoDB data for caching
def hash_documents(docs):
    doc_str = "".join([str(doc) for doc in docs])
    return hashlib.md5(doc_str.encode()).hexdigest()

@st.cache_resource(show_spinner=False)
def get_cached_faiss_index(documents_hash):
    # Fetch documents again inside cache to bind them to hash
    raw_docs = get_documents()
    langchain_docs = [Document(page_content=str(doc)) for doc in raw_docs]

    # Split docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    split_docs = splitter.split_documents(langchain_docs)

    # Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    return vectorstore.as_retriever(search_kwargs={"k": 10})

# 🔄 On every interaction
def get_rag_chain():
    raw_docs = get_documents()
    docs_hash = hash_documents(raw_docs)
    retriever = get_cached_faiss_index(docs_hash)

    llm = GradioMistralLLM(space_id="aiqcamp/Llama-4-Maverick-17B-search")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the user's question based only on the given context.
Avoid any repetition or alternative responses.

Context:
{context}

Question: {question}
Answer:
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain
