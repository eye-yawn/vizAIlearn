import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader 
import requests
from bs4 import BeautifulSoup

# title and intro
st.set_page_config(page_title="Summary", page_icon="📚")

st.markdown("# Visual Text Summary")
st.write(
    """As the title of the app suggests, this feature allows for instantaneous
    summarization of any inputted texts or PDFs, coupled with a visual representation of those main ideas."""
)

st.write('''\nBelow, an example has already been filled in. You can toggle the number of summaries 
         and watch the magic happen, or you can enter your own text or upload a file to try once you're ready. 
        ''')

##########################################################
# summm and viz methods
##########################################################
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

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

@st.cache_resource
def downloadpunkt():
    nltk.download('punkt')

downloadpunkt()

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

@st.cache_resource
def gen_lex_rank_summary (input_text, num_summaries,language):
  parser = PlaintextParser.from_string(input_text,Tokenizer(language))
  summarizer = LexRankSummarizer()
  summary_lexrank = summarizer(parser.document, num_summaries)
  return summary_lexrank

@st.cache_resource
def gen_luhn_summary (input_text, num_summaries,language):
  parser = PlaintextParser.from_string(input_text,Tokenizer(language))
  summarizer_luhn = LuhnSummarizer()
  summary_luhn =summarizer_luhn(parser.document,num_summaries)
  return summary_luhn

@st.cache_resource
def gen_lsa_summary (input_text, num_summaries,language):
  parser = PlaintextParser.from_string(input_text,Tokenizer(language))
  summarizer_lsa = LsaSummarizer()
  summary_lsa = summary_lsa =summarizer_lsa(parser.document,num_summaries)
  return summary_lsa

@st.cache_resource
def gen_lsa_summary_stopwords (input_text, num_summaries,language):
  parser = PlaintextParser.from_string(input_text,Tokenizer(language))
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

########################################
# Visualization
########################################

from streamlit_d3graph import d3graph
import pandas as pd
from d3blocks import D3Blocks
import streamlit.components.v1 as components

@st.cache_data
def d3_visualize(summary):
    df = pd.DataFrame({
        'source':['Summaries']*len(summary),
        'target': summary,
        'weight': 1
    })
    d3 = D3Blocks(chart='tree')
    d3.set_node_properties(df)
    d3.set_edge_properties(df)
    d3html = d3.show(filepath=None)
    components.html(d3html, height=800,width=600)

# Summarization functions in one

def summarize(input_text,num_sum,language):
  # Generate ensemble summary (t5 is excluded for now although it can be added by adding ",summary_t5_sentences" as a parameter)
  summary_lexrank = gen_lex_rank_summary(input_text,num_sum,language)
  summary_luhn = gen_luhn_summary(input_text,num_sum,language)
  summary_lsa = gen_lsa_summary(input_text,num_sum,language)
  summary_lsa_stopwords = gen_lsa_summary_stopwords(input_text,num_sum,language)
  #print('Generating the Best of the Breed Summaries')
  final_summary = ensemble_summary(num_sum, summary_lexrank, summary_luhn, summary_lsa, summary_lsa_stopwords)
  return final_summary

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

num_sum = st.number_input("Enter how many sentences you want to summarize the input into (ex. 5):", min_value=1, max_value=15, step=1)
language = st.selectbox('Enter the language of the text you wish to generate the summary for:', ('en', 'fr', 'es', 'chi'))

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

#  each chunk based on the selected option and the inputs collected above
for chunk in chunks:  
    final_summary = summarize(chunk, num_sum, language)
    # Display the final summary
    st.write('Summaries:')
    st.markdown("\n".join(f"- {item}" for item in final_summary))
    st.subheader('Tree diagram of the summaries')
    st.caption('Hover over a node or pan to the right to read the full sentence.')
    d3_visualize(final_summary)
