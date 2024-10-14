import streamlit as st
from embedding import *
import time
from langchain.chains.question_answering import load_qa_chain
from embedding import chatmodel
from langchain.chains import RetrievalQA

def QnA(index,question,retriever):

    query_vector = create_embedding(question)
    matches = index.query(vector=query_vector[0], top_k=4, include_metadata=True)
    # lst=[]
    # if matches['matches']:
    #     for match in matches['matches']:
    #         lst.append(match['metadata']['text'])
    # else:
    #     st.write("No matches found")

    qa_chain = RetrievalQA.from_chain_type(
        llm=chatmodel,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain({"query": question})
    answer = result.get('result', "I'm sorry, I couldn't generate an answer.")

    st.write(answer)
