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
#from unstructured.partition.pdf import partition_pdf

url = "https://www.fmcsa.dot.gov/sites/fmcsa.dot.gov/files/2022-04/FMCSA-HOS-395-DRIVERS-GUIDE-TO-HOS%282022-04-28%29_0.pdf"

os.environ["COHERE_API_KEY"] = getpass.getpass()
# pdf = (
#     "C:\\Users\\abhir\\syncd-llm\\FMCSA-HOS-395-DRIVERS-GUIDE-TO-HOS(2022-04-28)_0.pdf"
# )
# print("about to partition pdf")
# elements_fast = partition_pdf(pdf, strategy="fast")
# print(elements_fast[2])
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


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")




vector_store = FAISS.from_documents(documents, embedding_model)


model = Cohere(model="command", max_tokens=512, temperature=0.75)

st.title("Logistics Copilot")
user_query = st.text_input("Enter your question:")

if user_query:
    # Embed the user query
    query_embedding = embedding_model.embed_query(user_query)

    # Find similar transcripts from the vector store
    result = vector_store.similarity_search(user_query, k=4)
    st.write(result)
    st.write(result[0])
    # summary_prompt = f"Summarize the following transcript in two or three sentences: {result[0]}"
    # transcript_summary_response = model(summary_prompt)
    conversation_prompt = f"""The user asked: {user_query}
    Here's the relevant source texts: {result}
    Can you provide an answer to the user's question using the relevant source text.
    '"""
    cohere_response = model(conversation_prompt)
    st.write(cohere_response)
