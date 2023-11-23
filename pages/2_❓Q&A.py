import streamlit as st
import os
import requests

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from io import BytesIO
import fitz
from bs4 import BeautifulSoup

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

# title and intro
st.set_page_config(page_title="Q&A", page_icon="❓")

st.markdown("# Interactive Q&A")
st.write(
    """Have a question about the text but can't find the answer. Try this interactive Q&A feature that pulls responses
    directly from the text so you can be sure that the answer is reliable."""
)

##########################################################
# Q&A methods
##########################################################
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

# Chunking text
def chunk_text(text, max_chunk_size=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) <= max_chunk_size:
          current_length += len(word) + 1
          current_chunk.append(word)
        else:
            if len(current_chunk) > 0:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
    chunks.append(' '.join(current_chunk))
    return chunks

def createEmbeddings(chunked_docs):
    # Create embeddings and store them in a FAISS vector store
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunked_docs, embedder)
    return vector_store

@st.cache_resource
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

def get_user_input():
    source_option = st.radio("Choose input source", ["Enter Text", "Upload Text/PDF", "Input URL"])
    default_text = "Fungi, once considered plant-like organisms, are more closely related to animals than plants. Fungi are not capable of photosynthesis: they are heterotrophic because they use complex organic compounds as sources of energy and carbon. Fungi share a few other traits with animals. Their cell walls are composed of chitin, which is found in the exoskeletons of arthropods. Fungi produce a number of pigments, including melanin, also found in the hair and skin of animals. Like animals, fungi also store carbohydrates as glycogen. However, like bacteria, fungi absorb nutrients across the cell surface and act as decomposers, helping to recycle nutrients by breaking down organic materials to simple molecules. Some fungal organisms multiply only asexually, whereas others undergo both asexual reproduction and sexual reproduction with alternation of generations. Most fungi produce a large number of spores, which are haploid cells that can undergo mitosis to form multicellular, haploid individuals."
    if source_option == 'Enter Text':
        user_input_text = st.text_area('Enter your text here:',default_text)
        return 'text', user_input_text, None, None, None
    elif source_option == 'Upload Text/PDF':
        user_input_file = st.file_uploader('Upload your text or PDF')
        if user_input_file:
            if user_input_file.name.endswith('.pdf'):
                page_start = st.number_input("Enter the starting page number:", min_value=1, step=1)
                page_end = st.number_input("Enter the ending page number: \n Leave blank if starting and ending page numbers are the same.", min_value=page_start, step=1)
                section = st.text_input("Enter the section (example: Introduction, Conclusion etc. Leave blank to retrieve all):")
                return 'pdf', user_input_file, page_start, page_end, section
            else:
                return 'text', user_input_file, None, None, None
        else:
            st.error("ERROR: Please upload a file first.",icon="⚠️")
            return None, None, None, None, None
    elif source_option == 'Input URL':
        url = st.text_input('Enter the URL:')
        if url:
            response = requests.get(url)
            if url.endswith('.pdf'):
                return 'pdf', BytesIO(response.content), None, None, None
            else:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text from the webpage
                for script in soup(["script", "style"]):
                    script.decompose()
                text = " ".join(t.strip() for t in soup.stripped_strings)
                return 'text', text, None, None, None
        else:
            st.error("ERROR: Please enter a URL first.",icon="⚠️")
            return None, None, None, None, None    
    return None, None, None, None, None

##########################################################
# streamlit side
##########################################################
# Get user input
input_type, data, page_start, page_end, section = get_user_input()

if input_type == 'text':
    chunks = [data]

elif input_type == 'pdf':
    extracted_text = ""
    with fitz.open(stream=data.getvalue(), filetype="pdf") as doc:
        for page in doc:
            extracted_text += page.get_text()
        if section:
            extracted_text = extracted_text.split(section)[1] if section in extracted_text else ""
        if page_start and page_end:
            extracted_text = extracted_text.split("\n")[page_start-1:page_end]
            extracted_text = " ".join(extracted_text)
    chunks = [extracted_text]

# Process each chunk based on the selected option and the inputs collected above
for chunk in chunks:  
    question = st.text_area("Ask a question about your text or PDF:", value = "What is chitin?", height = 20)
    prepared_docs = prepareDocument(chunk)
    LOCAL_vector_store = createEmbeddings(prepared_docs)
    
    token = st.secrets["huggingfacetoken"]
    if token:
        chain = loadLLMModel(token)
        LOCAL_response = askQuestions(LOCAL_vector_store, chain, question)        
        st.text_area(label="Here's the answer:", value=LOCAL_response, height=100)
