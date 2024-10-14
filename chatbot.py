import os
import streamlit as st
from dotenv import load_dotenv
from langchain_fireworks import FireworksEmbeddings
from langchain_fireworks import ChatFireworks
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from fileops import *
from embedding import *

# load_dotenv()

# fireworks_api_key=os.getenv('FIREWORKS_API_KEY')
pinecone_api_key=os.getenv('PINECONE_API_KEY')

pc=Pinecone(api_key=pinecone_api_key)

# Setting up pinecone index

index_name = "vector-index"  # change if desired
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
index = pc.Index(index_name)




# # Defining models
#
# embmodel = FireworksEmbeddings(
#     model="nomic-ai/nomic-embed-text-v1.5",
# )
# chatmodel=ChatFireworks(
#         api_key=fireworks_api_key,
#         model="accounts/fireworks/models/llama-v3-70b-instruct",
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#     )


# # function for text extraction
#
# def text_extraction(file,typefile):
#     text=''
#     if typefile=='pdf':
#         pdf=PdfReader(file)
#         for page in pdf.pages:
#             text+= page.extract_text()
#         return(text)
#
#     elif typefile in ('jpg','jpeg','png'):
#         pass



# function for splitting text and converting it into chunks

# def text_splitting(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators="\n",
#         chunk_size=500,
#         chunk_overlap=100,
#         length_function=len
#     )
#     text_chunks=text_splitter.split_text(text)
#     return(text_chunks)


# function for generating embeddings

# def create_embedding(text_chunks):
#     embeddings=embmodel.embed_documents(text_chunks)
#     st.write(embeddings)
#     return embeddings


# initializing vector store

vector_store = PineconeVectorStore(index=index, embedding=embmodel)


def store_vectors(text_chunks,embeddings):
    batch_size = 100  # Adjust based on your Pinecone plan and requirements

    for i in range(0, len(text_chunks), batch_size):
        batch_texts = text_chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]


        vectors = [
            (str(j + i), embedding, {"text": text})
            for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings))
        ]

        index.upsert(vectors=vectors)










# main interface

with st.sidebar:
    st.header('Welcome to the ChatApp')
    file=st.text_input('Please provide path for the file: ')
    file=file[1:]
    file=file[:-1]
    btn=st.button('Submit')


if btn:
    if file:
        st.write(file)
        lastindex=file.rindex('.')
        typefile = file[lastindex + 1:]
        st.write("File type is: ",typefile)
        text=text_extraction(file,typefile)
        st.text_area('',text,height=350)
        text_chunks=text_splitting(text)
        #st.write(text_chunks)
        embeddings=create_embedding(list(text_chunks))
        st.write(embeddings)
        store_vectors(list(text_chunks),list(embeddings))




    else:
            errorcontainer=st.empty()
            errorcontainer.error('File not found.Please enter path correctly!!')
            time.sleep(2)
            errorcontainer.empty()

