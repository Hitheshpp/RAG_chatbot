from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_community.llms import HuggingFaceHub
#from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from gradio_mistral_llm import GradioMistralLLM

# Step 1: Load and split text
loader = TextLoader("college_data.txt.md", encoding="utf-8")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
documents = splitter.split_documents(pages)

# Step 2: Embedding & Vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

#llm = GradioMistralLLM("https://hysts-mistral-7b.hf.space/")
llm = GradioMistralLLM("hysts/mistral-7b")

# Step 4: RAG Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
