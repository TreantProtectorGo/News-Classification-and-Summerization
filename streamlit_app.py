import streamlit as st
import random
import time
import numpy as np
from io import StringIO
from streamlit_news import load_data
from streamlit_news import getCategory
from streamlit_news import recommendNews

count = 0
news_df = load_data()

# Page title
st.set_page_config(layout="wide", page_title='News Classification and Summerization')
st.title('News Classification and Summerization')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

uploaded_file = st.file_uploader("Choose a image or video file", type=["jpg","jpeg","png","mp4","mkv","mov","avi"])
if uploaded_file is not None:
    # To read file as bytes:s
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

# Accept user input
if prompt := st.chat_input("Prase your news here"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        greetings = ["hi", "hello", "hey", "helloo", "hellooo", "g morining", "gmorning", "good morning", "morning", "good day", "good afternoon", "good evening", "greetings", "greeting", "good to see you", "its good seeing you", "how are you", "how're you", "how are you doing", "how ya doin'", "how ya doin", "how is everything", "how is everything going", "how's everything going", "how is you", "how's you", "how are things", "how're things", "how is it going", "how's it going", "how's it goin'", "how's it goin", "how is life been treating you", "how's life been treating you", "how have you been", "how've you been", "what is up", "what's up", "what is cracking", "what's cracking", "what is good", "what's good", "what is happening", "what's happening", "what is new", "what's new", "what is neww", "g’day", "howdy", "what can you do"]
        if prompt.lower() in greetings:
                assistant_response = random.choice(
                    [
                        "Welcome! How can I assist you today as a News Classification and Summarization bot?",
                        "Hello there! Is there any news-related query or topic you'd like assistance with?",
                        "Looking for help with news classification and summarization? I'm here to assist you!"
                    ]
                )
        else:
            extra_prediction = getCategory(prompt, news_df, count)
            assistant_response = extra_prediction
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.1)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# Using "with" notation
with st.sidebar:
    options = st.multiselect(
        'Select news category',
        ['Technology', 'Sports', 'World', 'Business', 'Politics', 'Entertainment', 'Science', 'Travel', 'Fashion', 'Education'],
        [], max_selections=5)
    
    if options == []:
        st.write('Please select a category')
    else: 
        recommended_news = recommendNews(options)
        st.write('Recommended News:')   
        for news in recommended_news:
            st.write('Title ' + str(recommended_news[0]))
            st.write('Category: ' + str(recommended_news[1]))
            st.write('Description: ' + str(recommended_news[2]))