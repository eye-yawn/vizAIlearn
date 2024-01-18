import streamlit as st
from PIL import Image

st.set_page_config(page_title="VizAILearn", page_icon="🎓")

st.title("Viz-AI-Learn: The Visual Learning Platform of the Future")
st.write("Tired of sifting through endless pages of text? Worried about an upcoming test on several confusing concepts? \n\n This AI-powered tool is here to make your life easier. It can convert long-winded textbook pages into concise summaries while adding a variety of digestible visualizations to convey key ideas. \n\nOnce you're feeling comfortable, try the Q&A feature to instantly generate reliable answers to your practice questions and master knowledge. \nSay goodbye to stressful studying and hello to smarter, faster learning of the future with Viz-AI-Learn")
st.write('\nTo start using this app, click on one of the pages to the left. \nHappy Learning!\n')
st.caption('Created by Aayaan Ahmed -- LinkedIn: https://www.linkedin.com/in/aayaan-ahmed/')

image = Image.open('stressed_student (1).jpg')

col1,col2,col3,col4,col5 = st.columns(5)
with col2:
    st.image(image, caption='The Guardian: "Many students are unable to concentrate long enough to finish their studies. Photograph: Alamy"', width =400)

# with st.sidebar.form(key ='Form1', clear_on_submit=True):
#     st.write("Feedback ")
#     stars = st.radio("Rate your experience on Viz-AI-Learn",[':star:',':star::star:',':star::star::star:',':star::star::star::star:',':star::star::star::star::star:'])
#     user_feedback = st.text_area("If you have any feedback or concerns please let me know here:")
#     submitbutton = st.form_submit_button(use_container_width=True)
#     if submitbutton:
#         st.write("Thank you for your feedback.")

import smtplib
from email.mime.text import MIMEText

with st.sidebar:
    st.markdown("**Feedback 💬**")
    email_sender = st.secrets["sendemail"]
    email_receiver = st.secrets["receiveremail"]
    
    # inputs
    stars = st.radio("Rate your experience on Viz-AI-Learn",[':star:',':star:'*2,':star:'*3,':star:'*4,':star:'*5])
    numstars = str(stars.count(":star:"))
    user_feedback = st.text_area("If you have any feedback or concerns please let me know here:")
    body = numstars + " stars:\n\n" + user_feedback

    if st.button("Send"):
        try:
            msg = MIMEText(body)
            msg['From'] = email_sender
            msg['To'] = email_receiver

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(email_sender, st.secrets["password"])
            server.sendmail(email_sender, email_receiver, msg.as_string())
            server.quit()

            st.success('Thank you for your feedback! 🚀')
        except Exception as e:
            st.error(f"Failed to send email: {e}")

