import streamlit as st
from embedding import *
import time

def QnA(index,question):

    query_vector = create_embedding(question)
    matches = index.query(vector=query_vector[0], top_k=2, include_metadata=True)

    st.write(matches)

