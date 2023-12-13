import streamlit as st
import random
import time

import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import string
from nltk.corpus import stopwords
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Page title
st.set_page_config(layout="wide", page_title='News Classification and Summerization')
st.title('News Classification and Summerization')

@st.cache_data
def load_data():
    categories = ['technology', 'sports', 'world', 'business', 'politics', 'entertainment', 'science', 'travel', 'fashion', 'education']

    title = []
    category = []
    content = []

    for categoryIndex in categories:
        for i in range(1, 101):
            # example
            # https://inshorts.com/api/en/search/trending_topics/business?page=4&type=NEWS_CATEGORY
            url = f'https://inshorts.com/api/en/search/trending_topics/{categoryIndex}?page={i}&type=NEWS_CATEGORY'
            response = requests.get(url)
            data = json.loads(response.text)

            for news in data['data']['news_list']:
                if 'news_obj' in news:
                    title.append(news['news_obj']['title'])
                    category.append(categoryIndex)
                    #category.append(news['news_obj']['category_names'])
                    content.append(news['news_obj']['content'])
                        
    news_df = pd.DataFrame({'title': title, 'content': content, 'category': category})

    news_df['title'] = news_df['title'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    news_df['title'] = news_df['title'].apply(lambda x: "".join([char for char in x if char not in string.punctuation]))
    nltk.download('stopwords')
    stop = stopwords.words('english')
    news_df['title'] = news_df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    news_df['title'] = news_df['title'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    nltk.download('wordnet')
    lem = nltk.WordNetLemmatizer()
    news_df['title'][:10].apply(lambda x: " ".join([lem.lemmatize(word, 'v') for word in x.split()]))

    return news_df

def getCategory(input_value):
    # Split the data into training and test sets
    training_corpus, test_corpus = train_test_split(news_df, test_size=0.2, random_state=42)

    # Preparing data for SVM model (using the same training_corpus, test_corpus from 
    # naive bayes example)
    train_data = []
    train_labels = []
    for index, row in training_corpus.iterrows():
        train_data.append(row['title'])
        train_labels.append(row['category'])

    test_data = []
    test_labels = []
    for index, row in test_corpus.iterrows():
        test_data.append(row['title'])
        test_labels.append(row['category'])

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.8)

    # Train the feature vectors
    train_vectors = vectorizer.fit_transform(train_data)

    # Apply model on test data
    test_vectors = vectorizer.transform(test_data)

    # Perform classification with SVM, kernel=linear
    model = svm.SVC(kernel='linear')
    model.fit(train_vectors, train_labels)
    prediction = model.predict(test_vectors)
    
    # Additional testing with new data
    extra_test_data = input_value
    # Convert extra test data to feature vectors
    extra_test_vectors = vectorizer.transform([extra_test_data])
    # Make predictions on extra test data
    extra_prediction = model.predict(extra_test_vectors)

    return str(extra_prediction[0])

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Select news category",
    ("Email", "Home phone", "Mobile phone")
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )

# txt = st.text_area( 
#     "Parse your news here",
#     "",
#     height=200)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
            extra_prediction = None
            if extra_prediction is None:
                with st.spinner('Wait for crawling ...'):
                    time.sleep(30)
                extra_prediction = getCategory(prompt)
                assistant_response = extra_prediction
            else:
                extra_prediction = getCategory(prompt)
                assistant_response = extra_prediction
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

