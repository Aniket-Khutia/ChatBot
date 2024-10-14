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
from vecdb import *

load_dotenv()

# fireworks_api_key=os.getenv('FIREWORKS_API_KEY')
# pinecone_api_key=os.getenv('PINECONE_API_KEY')
#
# pc=Pinecone(api_key=pinecone_api_key)


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
        embeddings=create_embedding(list(text_chunks))  # gettiong the embeddings and dimension of the embeddings
        embeds=embeddings[0]    # for getting the embeddings
        dimension=embeddings[1]  # for getting the dimension of vector store
        index=setindex(dimension) # for getting index
        vecdbinit(index,embmodel) # for initializing pinecone vector store





        # st.write(embeds[0])
        # st.write(dimension)  # Displaying embedding and dimension of each embedding

        store_vectors(list(text_chunks),list(embeds),index)
        # query='What is the name of the sender?'
        # query_vector=create_embedding(list(query))
        # matches=index.query(vector=query_vector,top_k=2)
        # st.write(matches)




    else:
            errorcontainer=st.empty()
            errorcontainer.error('File not found.Please enter path correctly!!')
            time.sleep(2)
            errorcontainer.empty()


