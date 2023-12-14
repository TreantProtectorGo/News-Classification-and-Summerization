import streamlit as st
import random
import time
import numpy as np
import imghdr
import pandas as pd
from io import StringIO
from streamlit_news import load_data
from streamlit_news import getCategory
from streamlit_news import recommendNews
from streamlit_news import imageDecode
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

count = 0
news_df = load_data()

# Page title
st.set_page_config(layout="wide", page_title='News Classification and Summerization')
st.title('News Classification and Summerization')


def summarizer(extra_prediction):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    ARTICLE = extra_prediction
    summary = summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)
    return summary

def summarize_content(content):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(content, max_length=80, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def generate_transcript(id):
	transcript = YouTubeTranscriptApi.get_transcript(id)
	script = ""

	for text in transcript:
		t = text["text"]
		if t != '[Music]':
			script += t + " "
		
	return script, len(script.split())

col1, col2 = st.columns([0.5, 0.5])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with col2:
    uploaded_file = st.file_uploader("Choose a image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        isImage = imghdr.what(uploaded_file)
        if isImage is not None:
            output_text = imageDecode(uploaded_file)
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write(output_text)

        # # To read file as bytes: 
        # bytes_data = uploaded_file.getvalue()
        
        # # To convert to a string based IO:
        # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)

        # # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)

        # # Can be used wherever a "file-like" object is accepted:
        # dataframe = pd.read_csv(uploaded_file)
        # st.write(dataframe)

with col1:
    taskOptions = st.multiselect(
        'Select a task',
        ['Classification', 'Summarization'],
        ['Classification', 'Summarization'])
    
# Accept user input
if prompt := st.chat_input("Prase your news or youtube url here"):
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
                        "Looking for help with news classifiv  v b v cation and summarization? I'm here to assist you!"
                    ]
                )
        else:  
            if len(prompt.split()) > 1024:
                assistant_response = "Please enter a shorter news article"
            else:
                if 'youtube.com' in prompt:
                    id = prompt.split('=')[1]
                    transcript, no_of_words = generate_transcript(id)
                    prompt = transcript  
                if taskOptions == ['Classification']:
                    extra_prediction = getCategory(prompt, news_df, count)
                    assistant_response = extra_prediction
                elif taskOptions == ['Summarization']:
                    summarizer_response = summarizer(prompt)
                    assistant_response = summarizer_response[0]['summary_text']
                elif not taskOptions:
                    assistant_response = "Please select a task"
                else:
                    if len(prompt.split()) < 50:
                        assistant_response = "Please enter a longer news article"
                    else:
                        extra_prediction = getCategory(prompt, news_df, count)
                        assistant_response = extra_prediction
                        summarizer_response = summarizer(prompt)
                        assistant_response = 'Category: ' + extra_prediction + 'space' + summarizer_response[0]['summary_text']
        for chunk in assistant_response.split():
            full_response += chunk + " "
            # if 'space' in assistant_response:
            #     full_response += chunk.replace('space', '\n') + " " 
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
    
    if options:
        category_news = recommendNews(options)
        st.sidebar.write('Recommended News:')
        for category in options:
            st.sidebar.subheader(category)
            for news_title, news_url, news_content in category_news[category]:
                st.sidebar.markdown(f"Title - [{news_title}]({news_url})" if news_url != "URL not available" else news_title)
                # Display the summary
                try:
                    summary = summarize_content(news_content)
                    st.sidebar.write('Summary : ' + summary)
                except Exception as e:
                    st.sidebar.write("Error in summarization.")
