import streamlit as st
from PIL import Image

st.set_page_config(page_title="VizAIlearn", page_icon="🎓")

st.title("VizAIlearn - The Visual Learning Platform of the Future")
st.write("Tired of sifting through endless pages of text? Worried about an upcoming test on several confusing concepts? \n\n This AI-powered tool is here to make your life easier. It can convert long-winded textbook pages into concise summaries while adding a variety of digestible visualizations to convey key ideas. \n\nOnce you're feeling comfortable, try the Q&A feature to instantly generate applicable practice questions and master knowledge. \nSay goodbye to stressful studying and hello to smarter, faster learning of the future with Viz-AI-learn")
st.write('\nTo start using this app, click on one of the pages to the left. \nHappy Learning!\n')
st.caption('Developed by Aayaan Ahmed -- LinkedIn: https://www.linkedin.com/in/aayaan-ahmed/')

image = Image.open('stressed_student (1).jpg')

col1,col2,col3,col4,col5 = st.columns(5)
with col2:
    st.image(image, caption='The Guardian: "Many students are unable to concentrate long enough to finish their studies. Photograph: Alamy"', width =400)
