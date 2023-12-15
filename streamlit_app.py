import streamlit as st
import random
import time
import pyperclip
import numpy as np
import imghdr
import pandas as pd
from io import StringIO
from streamlit_news import load_data
from streamlit_news import getCategory
from streamlit_news import recommendNews
from streamlit_news import imageDecode
from transformers import pipeline
from newspaper import Article
from transformers import BartTokenizer, BartForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi

# Function to load the summarization model
def load_summarizer_model():
    return pipeline("summarization")

# Function to load BART model and tokenizer
def load_bart_model_and_tokenizer():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

# Check if the models and tokenizer are already loaded in the session state, otherwise load them
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = load_summarizer_model()
if 'tokenizer' not in st.session_state or 'model' not in st.session_state:
    st.session_state.tokenizer, st.session_state.model = load_bart_model_and_tokenizer()

# Assign the tokenizer, model, and summarizer from session state to variables
tokenizer = st.session_state.tokenizer
model = st.session_state.model
summarizer = st.session_state.summarizer

count = 0
news_df = load_data()



# Page title
st.set_page_config(layout="wide", page_title='News Classification and Summerization')
st.title('News Classification and Summerization')

def extract_news_content_title_and_summary(url):
    article = Article(url, language="en")
    try:
        article.download()
        article.parse()
        article.nlp()  # Perform natural language processing on the article
    except Exception as e:
        return None, None, None, f"Failed to download or parse article: {e}"
    
    return article.text, article.title, article.summary, None

def summarizer(article_text):
    # Truncate the input text if it's too long
    if len(article_text) > 1024:  # Adjust this limit as needed
        article_text = article_text[:1024]

    inputs = tokenizer(article_text, return_tensors='pt', truncation=True, max_length=1024)

    try:
        summary_ids = model.generate(inputs['input_ids'], max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return [{"summary_text": summary}]
    except IndexError as e:
        return [{"summary_text": f"Error during summarization: {str(e)}. The text might be too long."}]
    except Exception as e:
        return [{"summary_text": f"An error occurred: {str(e)}"}]


def summarize_content(content):
    try:
        summarizer_response = summarizer(content)
        return summarizer_response[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {e}"


def generate_transcript(id):
	transcript = YouTubeTranscriptApi.get_transcript(id)
	script = ""

	for text in transcript:
		t = text["text"]
		if t != '[Music]':
			script += t + " "
		
	return script, len(script.split())

col1, col2 = st.columns([0.5, 0.5])

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
                pyperclip.copy(output_text)
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
        with st.spinner('Processing...'):  # Display a spinner during processing
            assistant_response = ""
            news_title = ""
            newspaper_summary = ""
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
                # if len(prompt.split()) > 1024:
                #     assistant_response = "Please enter a shorter news article"
                # else:
                if 'youtube.com' in prompt:
                    id = prompt.split('=')[1]
                    transcript, no_of_words = generate_transcript(id)
                    prompt = transcript
                elif prompt.startswith('http') or prompt.startswith('www') or '.com' in prompt:
                    news_content, fetched_title, fetched_summary, error = extract_news_content_title_and_summary(prompt)
                    isNewsURL = True
                    if error:
                        assistant_response = error
                    elif news_content:
                        prompt = news_content
                        news_title = fetched_title
                        newspaper_summary = fetched_summary
                if taskOptions == ['Classification']:
                        extra_prediction = getCategory(prompt, news_df, count)
                        assistant_response = extra_prediction
                elif taskOptions == ['Summarization']:
                    summarizer_response = summarizer(prompt)
                    assistant_response = 'Summary: ' + summarizer_response[0]['summary_text']
                elif not taskOptions:
                    assistant_response = "Please select a task"
                else:   
                    if len(prompt.split()) >= 50:
                        extra_prediction = getCategory(prompt, news_df, count)
                        assistant_response = extra_prediction
                        summarizer_response = summarizer(prompt)
                        if "Error" in summarizer_response[0]["summary_text"] or "Text too short" in summarizer_response[0]["summary_text"]:
                            st.warning(summarizer_response[0]["summary_text"])
                        else:
                            assistant_response = f'Summary: {summarizer_response[0]["summary_text"]}'
                        assistant_response = 'Category: ' + extra_prediction + '\n\n' + 'Summary: ' + summarizer_response[0]['summary_text']
                    else:
                        if isNewsURL:
                            extra_prediction = getCategory(prompt, news_df, count)
                            assistant_response = extra_prediction
                            summarizer_response = summarizer(prompt)
                            if "Error" in summarizer_response[0]["summary_text"] or "Text too short" in summarizer_response[0]["summary_text"]:
                                st.warning(summarizer_response[0]["summary_text"])
                            else:
                                assistant_response = f'Summary: {summarizer_response[0]["summary_text"]}'
                            assistant_response = 'Category: ' + extra_prediction + '\n\n' + 'Summary: ' + summarizer_response[0]['summary_text']
                            isNewsURL = False
                        else:
                            assistant_response = "Please enter a longer news article"
            st.success('Done!')  # Indicate that processing is complete
            st.markdown(assistant_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
