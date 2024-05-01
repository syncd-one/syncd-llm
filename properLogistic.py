import streamlit as st
import langchain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
import os
import getpass
from langchain_community.llms import Cohere
from langchain_core.messages import HumanMessage
from bs4 import BeautifulSoup
import requests
import nltk
import pandas as pd

# Function to load documents
@st.cache_data
def load_documents():
    documents = []
    hazardous_materials = "hazardous_materials1.txt"
    cfr49text = "cfr49text.txt"
    
    with open(hazardous_materials, "r", encoding="utf8") as file:
        hazardous_materials_text = file.read()
        lines = hazardous_materials_text.split("\n")
        for line in lines:
            documents.append(Document(page_content=line))

    with open(cfr49text, "r", encoding="utf8") as file:
        cfr49text_text = file.read()
        lines = cfr49text_text.split("\n")
        for line in lines:
            documents.append(Document(page_content=line))
    
    return documents

# Function to load embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to load vector store
@st.cache_data
def load_vector_store(_documents, _embedding_model):
    return FAISS.from_documents(_documents, _embedding_model)

# Function to load Cohere model
@st.cache_resource
def load_cohere_model():
    os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
    return Cohere(model="command", max_tokens=512, temperature=0.75)

# Load data
documents = load_documents()
embedding_model = load_embedding_model()
vector_store = load_vector_store(documents, embedding_model)
cohere_model = load_cohere_model()

st.title("Logistics Copilot")
user_query = st.text_input("Enter your question:")

if user_query:
    # Embed the user query
    query_embedding = embedding_model.embed_query(user_query)

    # Find similar transcripts from the vector store
    result = vector_store.similarity_search(user_query, k=4)
    st.write(result)

    conversation_prompt = f"""The user asked: {user_query}
    Here's the relevant source texts: {result}
    Can you provide a professional and concise answer to the user's question using the relevant source text.
    '"""
    cohere_response = cohere_model(conversation_prompt)
    st.write(cohere_response)
