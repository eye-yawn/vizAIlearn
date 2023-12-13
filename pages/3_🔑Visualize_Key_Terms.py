import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader 
import requests
from bs4 import BeautifulSoup

# title and intro
st.set_page_config(page_title="Key Terms", page_icon="🔑")

st.markdown("# Visualize Key Terms")
st.write(
    """Extract key terms from your inputted text, visualize them in a tree diagram, and sort 
    them into named entities (which are common categories that often used in AI and NLP programs)."""
)

st.write('''Some common named entities: 
\norg = Organization
\nper = Person
\neve = Event
\nnat = Natural Phenomenon
\nYou can learn more about Named Entity Recognition (what this process is called) here: https://www.geeksforgeeks.org/named-entity-recognition/''')

##########################################################
# key terms and viz methods
##########################################################
import spacy
from spacy import displacy
import matplotlib.pyplot as plt

# Load spaCy's English model
@st.cache_resource
def loadenglishmodel():
    nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = loadenglishmodel()

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

##########################################################
# streamlit side
##########################################################
def get_user_input():
    source_option = st.radio("Choose input source", ["Enter Text", "Upload Text/PDF", "Input URL"])
    default_text = "Fungi, once considered plant-like organisms, are more closely related to animals than plants. Fungi are not capable of photosynthesis: they are heterotrophic because they use complex organic compounds as sources of energy and carbon. Fungi share a few other traits with animals. Their cell walls are composed of chitin, which is found in the exoskeletons of arthropods. Fungi produce a number of pigments, including melanin, also found in the hair and skin of animals. Like animals, fungi also store carbohydrates as glycogen. However, like bacteria, fungi absorb nutrients across the cell surface and act as decomposers, helping to recycle nutrients by breaking down organic materials to simple molecules. Some fungal organisms multiply only asexually, whereas others undergo both asexual reproduction and sexual reproduction with alternation of generations. Most fungi produce a large number of spores, which are haploid cells that can undergo mitosis to form multicellular, haploid individuals."
    if source_option == 'Enter Text':
        user_input_text = st.text_area('Enter your text here:',default_text)
        return 'text', user_input_text, None, None
    elif source_option == 'Upload Text/PDF':
        user_input_file = st.file_uploader("Upload pdf files", type=["pdf"])
        if user_input_file:
            if user_input_file.name.endswith('.pdf'):
                page_start = st.number_input("Enter the starting page number:", min_value=1, max_value=3000, step=1)
                page_end = st.number_input("Enter the ending page number (leave blank if starting and ending page numbers are the same):", min_value=page_start, max_value=3000, step=1)
                return 'pdf', user_input_file, page_start, page_end
            else:
                return 'text', user_input_file, None, None
        else:
            st.error("ERROR: Please upload a file first.",icon="⚠️")
            return None, None, None, None
    elif source_option == 'Input URL':
        url = st.text_input('Enter the URL:')
        if url:
            response = requests.get(url)
            if url.endswith('.pdf'):
                page_start = st.number_input("Enter the starting page number:", min_value=1, max_value=3000, step=1)
                page_end = st.number_input("Enter the ending page number (leave blank if starting and ending page numbers are the same):", min_value=page_start, max_value=3000, step=1)
                return 'pdf', BytesIO(response.content), page_start, page_end
            else:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text from the webpage
                for script in soup(["script", "style"]):
                    script.decompose()
                text = " ".join(t.strip() for t in soup.stripped_strings)
                return 'text', text, None, None
        else:
            st.error("ERROR: Please enter a URL first.",icon="⚠️")
            return None, None, None, None    
    return None, None, None, None

# Get user input
input_type, data, page_start, page_end = get_user_input()

if input_type == 'text':
    chunks = [data]
elif input_type == 'pdf':         
    text = ''
    reader = PdfReader(data)
    while (page_start <= page_end):
        page = reader.pages[page_start-1] 
        text += page.extract_text() 
        page_start += 1
    chunks = [text]
else:
    chunks = ''

# Process each chunk based on the selected option and the inputs collected above
for chunk in chunks: 
    preprocessed_text = preprocess_text(chunk)
    topics, entities, entity_labels = extract_topics_and_entities(preprocessed_text)
    
    # Generate and display tree diagram for topics
    visualize_topics_tree(topics)

    # Display entities with color-coding
    visualize_color_coded_entities(preprocessed_text) 
