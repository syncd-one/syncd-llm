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

url = "https://www.ecfr.gov/api/renderer/v1/content/enhanced/2024-03-13/title-49?subtitle=B&chapter=I&subchapter=C&part=172"
df = pd.read_html(url)
response = requests.get(url)

response.raise_for_status()

soup = BeautifulSoup(response.content, 'html.parser')

text_data = []
table_data = []
for p in soup.find_all('p'):
    text_data.append(p.text.strip())
for table in df:
    table_data.append(table.to_string())

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

documents = []
for text in text_data:
    documents.append(Document(page_content=text))

for table in table_data:
    documents.append(Document(page_content=table))

vector_store = FAISS.from_documents(documents, embedding_model)

os.environ["COHERE_API_KEY"] = getpass.getpass()

model = Cohere(model="command", max_tokens=512, temperature = 0.75)

st.title("Logistics Copilot")
user_query = st.text_input("Enter your question:")

if user_query:
    # Embed the user query
    query_embedding = embedding_model.embed_query(user_query)
    print(type(query_embedding))

    # Find similar transcripts from the vector store
    result = vector_store.similarity_search(user_query, k=4)
    st.write(result)
    # summary_prompt = f"Summarize the following transcript in two or three sentences: {result[0]}"
    # transcript_summary_response = model(summary_prompt)
    conversation_prompt = f"""The user asked: {user_query}
    Here's the relevant source text: {result[0]}
    Can you provide a helpful and informative answer to the user as if you were an assistant with expertise of the CFR 49 part 172'"""
    cohere_response=model(conversation_prompt)
    st.write(cohere_response)