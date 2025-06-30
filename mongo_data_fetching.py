from pymongo import MongoClient
import streamlit as st

username = st.secrets["MONGO_USERNAME"]
password = st.secrets["MONGO_PASSWORD"]  # URL-encoded if needed
db_name = "college_db"
collection_name = "full_details"

# Connection string
connection_string = f"mongodb+srv://{username}:{password}@cluster0.flfg6iv.mongodb.net/?retryWrites=true&w=majority"

# Connect to MongoDB
client = MongoClient(connection_string)
db = client[db_name]
collection = db[collection_name]

# Fetch all documents
documents = collection.find()

# Print all documents
# for doc in documents:
#     print(doc)
