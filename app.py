import streamlit as st
from rag_chatbot import qa_chain

st.title("🎓 RAG Chatbot – College Info Assistant")

query = st.text_input("Ask a question about college information:")

if query:
    with st.spinner("Searching..."):
        result = qa_chain.invoke({"query": query})
    st.write("📄 Answer:")
    st.success(result)
