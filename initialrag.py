import streamlit as st
from sentence_transformers import SentenceTransformer
import langchain
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
import os

# **Step 0: You need to have pre-collected YouTube transcripts (replace '...' with file paths)**
video_links = ["https://www.youtube.com/watch?v=xT8oP0wy-A0", "https://www.youtube.com/watch?v=ZUN3AFNiEgc", "https://www.youtube.com/watch?v=8KtDLu4a-EM"]

if os.path.exists('transcripts'):
    print('Directory already exists')
else:
    os.mkdir('transcripts')

for video_link in video_links:
    video_id = video_link.split('=')[1]
    transcript_file = os.path.join('transcripts', video_id + '.txt')

    if not os.path.exists(transcript_file):  # Check if we already have the transcript
        print(f"Generating transcript for {video_id}")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        with open(transcript_file, 'w') as f:
            for line in transcript:
                f.write(f"{line['text']}\n")
# **Step 1: Load HuggingFace Embedding Model**

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# **Step 2: Load Transcripts and Create Embeddings**
def load_and_embed_transcripts(embedding_model, transcripts_dir):
    transcripts = []

    for file in os.listdir(transcripts_dir):
        if file.endswith('.txt'):
            video_id = file.split('.')[0]
            with open(os.path.join(transcripts_dir, file), 'r') as f:
                text = f.read()
                if not isinstance(text, str):
                    text = str(text)
                transcripts.append(Document(page_content = text, metadata = {'video_id': video_id})  )

    return transcripts

# Load transcripts and create embeddings
transcripts = load_and_embed_transcripts(embedding_model, 'transcripts') 

# **Step 3: Create a FAISS Vector Store**
print((transcripts))
vector_store = FAISS.from_documents(transcripts, embedding_model)

# **Step 4: Streamlit Interface**
st.title("YouTube Transcript Search")
user_query = st.text_input("Enter your question:")

if user_query:
    # Embed the user query
    query_embedding = embedding_model.embed_query(user_query)
    print(type(query_embedding))

    # Find similar transcripts from the vector store
    result = vector_store.similarity_search(user_query)
    st.write(f"Found {len(result)} similar videos")
    st.write(result[0])

    # Display results
    # for video_id, similarity_score in result:
    #     st.write(f"Video ID: {video_id}")
    #     st.write(f"Transcript: {transcripts[video_id]}")
    #     st.write(f"Similarity Score: {similarity_score}")

