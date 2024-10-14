from langchain_fireworks import FireworksEmbeddings
from langchain_fireworks import ChatFireworks
from dotenv import load_dotenv
import os


fireworks_api_key=os.getenv('FIREWORKS_API_KEY')


# Defining models

embmodel = FireworksEmbeddings(
    model="nomic-ai/nomic-embed-text-v1.5",
)
chatmodel=ChatFireworks(
        api_key=fireworks_api_key,
        model="accounts/fireworks/models/llama-v3-70b-instruct",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

# Creating embeddings

def create_embedding(text_chunks):
    embeddings=embmodel.embed_documents(text_chunks)
    # st.write(embeddings)
    return [embeddings,len(embeddings[0])]
