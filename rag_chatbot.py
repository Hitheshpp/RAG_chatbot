from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from gradio_mistral_llm import GradioMistralLLM
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from mongo_data_fetching import documents

# Step 1: Load and split text
# loader = TextLoader("college_data.md", encoding="utf-8")
# pages = loader.load()

# loader = PyPDFLoader("college_data.pdf")
# pages = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
# documents = splitter.split_documents(pages)

#document_list = fetch_data_from_db()
document_list = documents
langchain_docs = [Document(page_content=str(doc)) for doc in document_list]

# # # Step 3: Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
documents = splitter.split_documents(langchain_docs)

# Step 2: Embedding & Vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

#llm = GradioMistralLLM("hysts/mistral-7b")
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

# Step 4: RAG Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",  
                                        retriever=retriever,
                                        chain_type_kwargs={"prompt": prompt},
                                        )
