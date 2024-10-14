from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import time
from langchain.chains import RetrievalQA

load_dotenv()
pinecone_api_key=os.getenv('PINECONE_API_KEY')

pc=Pinecone(api_key=pinecone_api_key)


# Setting up pinecone index
def setindex(dimension,index_name):
    flag=1
    index_name = index_name  # change if desired
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        flag=0
        pc.create_index(
            name=index_name,
            metric="cosine",
            dimension=dimension,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(4)
    index = pc.Index(index_name)
    return([index,flag])


# initializing vector store
def vecdbinit(index,embmodel):
    vector_store = PineconeVectorStore(index=index, embedding=embmodel)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever


# Storing chunks and vector embeddings
def store_vectors(text_chunks,embeddings,index):
    batch_size = 100  # Adjust based on your Pinecone plan and requirements

    for i in range(0, len(text_chunks), batch_size):
        batch_texts = text_chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]


        vectors = [
            (str(j + i), embedding, {"text": text})
            for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings))
        ]

        index.upsert(vectors=vectors)
        return