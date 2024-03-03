import cohere
import os
import hnswlib
import json
import json
import uuid
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

co =  cohere.Client(os.environ[''])

#https://txt.cohere.com/rag-chatbot/#step-by-step-guide

class Documents:

    def __init__(self, sources: List[Dict[str, str]]):
        self.sources = sources
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load()
        self.embed()
        self.index()


    def load(self) -> None:
        """
        Loads the documents from the sources and chunks the HTML content.
        """
        print("Loading documents...")

        for source in self.sources:
            elements = partition_html(url=source["url"])
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                self.docs.append(
                    {
                        "title": source["title"],
                        "text": str(chunk),
                        "url": source["url"],
                    }
                )

    def embed(self) -> None:
        """
        Embeds the documents using the Cohere API.
        """
        print("Embedding documents...")

        batch_size = 90
        self.docs_len = len(self.docs)

        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
		              texts=texts,
                      model="embed-english-v3.0",
                      input_type="search_document"
	 		).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval.
        """
        print("Indexing documents...")

        self.index = hnswlib.Index(space="ip", dim=1024)
        self.index.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.index.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.index.get_current_count()} documents.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves documents based on the given query.

        Parameters:
        query (str): The query to retrieve documents for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved  documents, with 'title', 'snippet', and 'url' keys.
        """
        docs_retrieved = []
        query_emb = co.embed(
                    texts=[query],
                    model="embed-english-v3.0",
                    input_type="search_query"
                    ).embeddings				    

        doc_ids = self.index.knn_query(query_emb, k=self.retrieve_top_k)[0][0]
        docs_to_rerank = []
        for doc_id in doc_ids:
            docs_to_rerank.append(self.docs[doc_id]["text"])

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v2.0",
        )

        doc_ids_reranked = []
        for result in rerank_results:
            doc_ids_reranked.append(doc_ids[result.index])

        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved

class Chatbot:

    def __init__(self, docs: Documents):
        self.docs = docs
        self.conversation_id = str(uuid.uuid4())

    def generate_response(self, message: str):
        response = co.chat(message=message, search_queries_only=True)

        if response.search_queries:
            print("Retrieving information...")
            documents = self.retrieve_docs(response)

            response = co.chat(
                message=message,
                documents=documents,
                conversation_id=self.conversation_id,
                stream=True,
            )
            for event in response:
                yield event
            yield response
        else:
            response = co.chat(
                message=message, 
                conversation_id=self.conversation_id, 
                stream=True
            )
            for event in response:
                yield event


    def retrieve_docs(self, response) -> List[Dict[str, str]]:
        """
        Retrieves documents based on the search queries in the response.

        Parameters:
        response: The response object containing search queries.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents.

        """
        # Get the query(s)
        queries = []
        for search_query in response.search_queries:
            queries.append(search_query["text"])

        # Retrieve documents for each query
        retrieved_docs = []
        for query in queries:
            retrieved_docs.extend(self.docs.retrieve(query))

        return retrieved_docs


class App:

    ...
    ...

    def run(self):

        while True:
            # Get the chatbot response
            response = self.chatbot.generate_response(message)

            # Print the chatbot response
            print("Chatbot:")
            citations_flag = False
            
            for event in response:
                stream_type = type(event).__name__
                
                # Text
                if stream_type == "StreamTextGeneration":
                    print(event.text, end="")

                # Citations
                if stream_type == "StreamCitationGeneration":
                    if not citations_flag:
                        print("\n\nCITATIONS:")
                        citations_flag = True
                    print(event.citations[0])
                    
                # Documents
                if citations_flag:
                    if stream_type == "StreamingChat":
                        print("\n\nDOCUMENTS:")
                        documents = [{'id': doc['id'],
                                    'text': doc['text'][:50] + '...',
                                    'title': doc['title'],
                                    'url': doc['url']} 
                                    for doc in event.documents]
                        for doc in documents:
                            print(doc)

            print(f"\n{'-'*100}\n")



sources = [
    {
        "title": "Text Embeddings", 
        "url": "https://docs.cohere.com/docs/text-embeddings"},
    {
        "title": "Similarity Between Words and Sentences", 
        "url": "https://docs.cohere.com/docs/similarity-between-words-and-sentences"},
    {
        "title": "The Attention Mechanism", 
        "url": "https://docs.cohere.com/docs/the-attention-mechanism"},
    {
        "title": "Transformer Models", 
        "url": "https://docs.cohere.com/docs/transformer-models"}   
]

documents = Documents(sources)
print(documents.docs)
chatbot = Chatbot(documents)
print(chatbot.docs.docs[0]["text"])
app = App(chatbot)
app.run()