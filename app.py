# import streamlit as st
# from rag_chatbot import qa_chain

# st.title("🎓 RAG Chatbot – College Info Assistant")

# query = st.text_input("Ask a question about college information:")

# if query:
#     with st.spinner("Searching..."):
#         result = qa_chain.invoke({"query": query})
#     st.write("📄 Answer:")
#     st.success(result)

import streamlit as st

# Load RAG chain only once
if "qa_chain" not in st.session_state:
    from rag_chatbot import qa_chain
    st.session_state.qa_chain = qa_chain

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# Style
st.markdown("""
    <style>
    .user-msg {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        text-align: right;
    }
    .bot-msg {
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        text-align: left;
    }
    .msg-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💬 RAG Chatbot – College Info Assistant")

# Show chat history
with st.container():
    st.markdown('<div class="msg-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role_class = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Form for user input
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Ask something about college...", key="input")
    submitted = st.form_submit_button("Send")
    if submitted and query:
        st.session_state.pending_user_input = query
        st.rerun()

# Handle user input and generate response
if st.session_state.pending_user_input:
    user_msg = st.session_state.pending_user_input
    st.session_state.messages.append({"role": "user", "content": user_msg})

    with st.spinner("Bot is thinking..."):
        result = st.session_state.qa_chain.invoke({"query": user_msg})
        answer = result.get("result", str(result))

    st.session_state.messages.append({"role": "bot", "content": answer})
    st.session_state.pending_user_input = None
    st.rerun()




