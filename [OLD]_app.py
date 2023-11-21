from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import re
import nltk

from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from collections import defaultdict
from rouge import Rouge

import plotly.graph_objects as go
#############
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}


#from IPython.display import display
#import nltk
nltk.download('punkt')
#formatting method for simpler main
def dashline():
  dashline = print('------------------')
  return dashline

def process_text(text):
  # Remove <pad> and </s> tags
  cleaned_text = re.sub(r'<pad>|</s>', '', text).strip()

  # Split the text by periods and filter out any empty sentences
  sentences = [sentence.strip() for sentence in cleaned_text.split('.') if sentence.strip()]

  # Join the sentences with newline characters
  return '\n'.join(sentences)

#############################
#Summary Methods
#############################

def gen_lex_rank_summary (input_text, num_summaries,language):
  parser = PlaintextParser.from_string(input_text,Tokenizer(language))
  print('Generating Summary with Lex Rank')
  summarizer = LexRankSummarizer()
  #Summarize the document with 5 sentences
  summary_lexrank = summarizer(parser.document, num_summaries)
  return summary_lexrank

def gen_luhn_summary (input_text, num_summaries,language):
  parser = PlaintextParser.from_string(input_text,Tokenizer(language))
  print('Generating Summary with Luhn')
  summarizer_luhn = LuhnSummarizer()
  summary_luhn =summarizer_luhn(parser.document,num_summaries)
  return summary_luhn

def gen_lsa_summary (input_text, num_summaries,language):
  parser = PlaintextParser.from_string(input_text,Tokenizer(language))
  print('Generating Summary with LSA')
  summarizer_lsa = LsaSummarizer()
  summary_lsa = summary_lsa =summarizer_lsa(parser.document,num_summaries)
  return summary_lsa

def gen_lsa_summary_stopwords (input_text, num_summaries,language):
  parser = PlaintextParser.from_string(input_text,Tokenizer(language))
  print('Summaries generated with Stopwords')
  summarizer_lsa2 = LsaSummarizer()
  summarizer_lsa2 = LsaSummarizer(Stemmer(language))
  summarizer_lsa2.stop_words = get_stop_words(language)
  summary_lsa_stopwords = list(summarizer_lsa2(parser.document, num_summaries))
  return summary_lsa_stopwords

#######################################################################
# Best of Breed ensemble
######################################################################
#Best of breed using ensemble  --- this will pick the best sentences from the outputs of all individual models
#Each summarizer "votes" for a sentence to be included in the final summary.
#For each sentence in the original document, we count how many summarizers included it in their summaries.
#We then rank the sentences based on the number of votes they received.
#The top num_summaries sentences (based on votes) are included in the ensemble summary.
#So, the ensemble summary consists of sentences that were deemed most relevant by multiple summarizers.
#The idea is that if multiple models agree on the importance of a sentence, it's more likely to be a key point from the original text.

# from collections import defaultdict
# import nltk
# from rouge import Rouge

# Ensemble method
def ensemble_summary(num_summaries, *summaries):
    votes = defaultdict(int)
    for summary in summaries:
        for sentence in summary:
            votes[str(sentence)] += 1

    # Sort sentences by votes and take the top 'num_summaries'
    sorted_sentences = sorted(votes.keys(), key=lambda k: votes[k], reverse=True)
    num_summaries=int(num_summaries)
    return sorted_sentences[:num_summaries]
# end of function

########################################
# Visualization
########################################

def format_as_hierarchy(summaries):
    """Formats the summarized bullet points in a hierarchical format."""
    # Indent each bullet point to represent it as a leaf in the hierarchy
    # sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summaries)
    hierarchy = {
      'name': 'summary',
      'children': [{'name': sentence} for sentence in summaries if sentence]
    }
    return hierarchy
########################################
import streamlit as st
import plotly.graph_objects as go
import textwrap

def visualize_network(data):
    fig = go.Figure()

    def add_node(node, x, y):
        wrapped_text = textwrap.fill(node['name'], width=30)
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            text=[wrapped_text],
            mode='markers+text',
            textposition="middle right",  # Adjust this for text positioning
            hoverinfo='text',
            textfont=dict(size=12, color="yellow"),
            marker=dict(symbol="circle", size=10, color='blue', line=dict(width=2.5, color='green'))
        ))

        if 'children' in node:
            num_children = len(node['children'])
            y_offset = 1.2
            start_y = y - ((num_children - 1) * y_offset) / 2
            for i, child in enumerate(node['children']):
                child_x = x + 0.25
                child_y = start_y + i * y_offset
                add_node(child, child_x, child_y)
                fig.add_trace(go.Scatter(
                    x=[x, child_x],
                    y=[y, child_y],
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    hoverinfo='none'
                ))

    add_node(data, 0, 0)

    fig.update_layout(
        title='Visual Summaries',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='black',
        width=900,
        height=600
    )

    st.plotly_chart(fig)


 

##########################################################
# Summarization functions in one
##########################################################
def summarize(input_text,num_sum,language):
  # Generate ensemble summary (t5 is excluded for now although it can be added by adding ",summary_t5_sentences" as a parameter)
  summary_lexrank = gen_lex_rank_summary(input_text,num_sum,language)
  summary_luhn = gen_luhn_summary(input_text,num_sum,language)
  summary_lsa = gen_lsa_summary(input_text,num_sum,language)
  summary_lsa_stopwords = gen_lsa_summary_stopwords(input_text,num_sum,language)
  #print('Generating the Best of the Breed Summaries')
  final_summary = ensemble_summary(num_sum, summary_lexrank, summary_luhn, summary_lsa, summary_lsa_stopwords)
  #st.write(type(final_summary))
  return final_summary

##########################################################
# Chunking text
##########################################################
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

##########################################################
# extracting topics and entitites
##########################################################
# Importing necessary libraries
import spacy
from spacy import displacy
import matplotlib.pyplot as plt

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Load spaCy's stopwords
from spacy.lang.en.stop_words import STOP_WORDS

def preprocess_text(text):
    """
    Preprocesses the input text - tokenizes, lemmatizes, and removes stopwords.
    """
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def extract_topics_and_entities(text):
    """
    Extracts topics and entities from the input text.
    """
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    entity_labels = [ent.label_ for ent in doc.ents]
    topics = [chunk.text for chunk in doc.noun_chunks if chunk.text not in STOP_WORDS]
    return topics, entities, entity_labels

import plotly.figure_factory as ff

import plotly.graph_objects as go
import networkx as nx

def visualize_topics_tree(topics):
    """
    Visualizes the key topics as a tree diagram.
    """
    if not topics or len(topics) < 2:
        st.warning("Insufficient topics found to visualize.")
        return

    # Construct a graph from the sequence of topics
    G = nx.DiGraph()
    for i in range(len(topics) - 1):
        G.add_edge(topics[i], topics[i+1])

    # Use a tree layout to position the nodes
    pos = nx.layout.planar_layout(G)
    for node, coords in pos.items():
        G.nodes[node]['pos'] = coords

    # Convert the positions to a format suitable for Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(size=10, color='skyblue'),
        text=topics,
        textposition='top center')

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(t=50, b=0, l=40, r=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Display using Streamlit
    st.plotly_chart(fig)

######################
import spacy
from spacy import displacy

def visualize_color_coded_entities(text):
    """
    Visualizes entities in the text with color-coding.
    """
    doc = nlp(text)

    # Convert the displacy output for Streamlit. 
    # This function takes in a spaCy Doc object and outputs the raw HTML for visualization.
    html = displacy.render(doc, style="ent")
    
    # Streamlit's method to display raw HTML
    st.write(html, unsafe_allow_html=True)

#############################################################################################
### QA using Langchain
###########################################################################################
import os
import requests

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

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


#############################################################################################
### Putting it all together in streamlit
###########################################################################################
#function to handle user-input

import streamlit as st
from io import BytesIO
import fitz
import requests
from bs4 import BeautifulSoup

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
                
    return None, None, None, None, None

############################
# Custom CSS for Streamlit
# Custom CSS for Streamlit Sidebar
custom_css = """
<style>
    /* Background for the entire sidebar */
    div[data-testid="stSidebar"][role="region"] {
        background-color: #4E5D6C;  /* Change to your desired color */
        color: white;  /* Change text color */
        width: 200px;  /* Set width of the sidebar */
    }

    /* Buttons and select boxes in the sidebar */
    div[data-testid="stSidebar"][role="region"] .stButton>button, 
    div[data-testid="stSidebar"][role="region"] .stTextInput>div>div>input,
    div[data-testid="stSidebar"][role="region"] .stSelectbox>div>div>select {
        background-color: #2F3B48;  /* Change to your desired color */
        color: yellow;  /* Change text color */
        border: none;  /* Remove default border */
    }
</style>
"""
# Inject the custom CSS into Streamlit
st.markdown(custom_css, unsafe_allow_html=True)

# Title and Styling
st.title("VizAIlearn - The Visual Learning Platform of the Future")
st.write("Tired of sifting through endless pages of text? Worried about an upcoming test on several confusing concepts? \n\n This AI-powered tool is here to make your life easier. It can convert long-winded textbook pages into concise summaries while adding a variety of digestible visualizations to convey key ideas. \n\nOnce you're feeling comfortable, try the Q&A feature to instantly generate applicable practice questions and master knowledge. \nSay goodbye to stressful studying and hello to smarter, faster learning of the future with Viz-AI-learn")
st.write('\nTo use this app, toggle the number of summaries with the sample text and watch the magic happen. \nAn example has already been filled in below, but you can enter your own text or upload a file once you are ready. \nHappy Learning!')

# Sidebar for feature selection
option = st.sidebar.selectbox(
    'Choose A Learning Feature',
    ('Visual Text Summary', 'Q&A from Text/PDF', 'Visualize Key Terms', 'Visual Timeline of an Event in History')
)

# Get user input
input_type, data, page_start, page_end, section = get_user_input()

# Collect user inputs for features
if option == "Visual Timeline of an Event in History":
      st.write("Coming Soon!")
elif option == 'Visual Text Summary':
    num_sum = st.number_input("Enter how many sentences you want to summarize the input into (ex. 5):", min_value=1, step=1)
    language = st.selectbox('Enter the language of the text you wish to generate the summary for:', ('en', 'fr', 'es', 'chi'))

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
    if option == 'Visual Text Summary':
        final_summary = summarize(chunk, num_sum, language)
        # Display the final summary
        st.write('Summaries:')
        st.markdown("\n".join(f"- {item}" for item in final_summary))
        #convert to hierarchial
        topic = format_as_hierarchy(final_summary)
        print(topic)
        # Call the function to visualize the data
        visualize_network(topic)

    elif option == "Visualize Key Terms":
        #topics, entities, entity_labels = extract_topics_and_entities(chunk)
        # Visualize key topics as a network graph
        #visualize_topics_as_network(topics)
        # Visualize named entities
        #visualize_named_entities(chunk)
        # Example usage:
        #text = "Apple Inc. is a technology company. Steve Jobs was one of its co-founders. The company is based in Cupertino."
        preprocessed_text = preprocess_text(chunk)
        topics, entities, entity_labels = extract_topics_and_entities(preprocessed_text)
    
         # Generate and display tree diagram for topics
        visualize_topics_tree(topics)

          # Display entities with color-coding
        visualize_color_coded_entities(preprocessed_text)

    elif option == "Q&A from Text/PDF":
      question = st.text_area("Ask your text/PDF a question", height=20)
      prepared_docs = prepareDocument(chunk)
      LOCAL_vector_store = createEmbeddings(prepared_docs)

      token = st.text_input("Enter HuggingFaceHub API Token", type='password')
      if token:
        chain = loadLLMModel(token)

        LOCAL_response = askQuestions(LOCAL_vector_store, chain, question)
        st.text_area(label="", value=LOCAL_response, height=100)

    
