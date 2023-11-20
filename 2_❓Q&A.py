import streamlit as st
import os
import requests

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

#methods
def createEmbeddings(chunked_docs):
    # Create embeddings and store them in a FAISS vector store
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunked_docs, embedder)
    return vector_store

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

def loadLLMModel(token):
    llm = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", 
                     model_kwargs={"temperature":0, "max_length":512},
                     huggingfacehub_api_token=token)

    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def askQuestions(vector_store, chain, question):
    # Ask a question using the QA chain
    similar_docs = vector_store.similarity_search(question)
    response = chain.run(input_documents=similar_docs, question=question)
    return response

def prepareDocument(chunk, metadata=None):
    return [Document(chunk, metadata)]

#streamlit side
st.set_page_config(page_title="Q&A", page_icon="❓")

st.markdown("# Interactive Q&A")
st. title("Interactive Q&A")
st.write(
    """Have a question about the text but can't find the answer. Try this interactive Q&A feature that pulls responses
    directly from the text so you can be sure that the answer is reliable."""
)

question = st.text_area("Ask your text/PDF a question", height=20)
prepared_docs = prepareDocument(chunk)
LOCAL_vector_store = createEmbeddings(prepared_docs)

token = st.text_input("Enter HuggingFaceHub API Token", type='password')
if token:
    chain = loadLLMModel(token)

LOCAL_response = askQuestions(LOCAL_vector_store, chain, question)
st.text_area(label="", value=LOCAL_response, height=100)
