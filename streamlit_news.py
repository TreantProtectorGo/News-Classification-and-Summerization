import requests
import ujson
import random
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

nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('wordnet')
lem = nltk.WordNetLemmatizer()

def load_data():
    title = []
    category = []
    content = []

    url = "https://raw.githubusercontent.com/TreantProtectorGo/News-Classification-and-Summerization/main/news.csv?raw=true"
    news_df = pd.read_csv(url)
    for index, row in news_df.iterrows():
        title.append(row['title'])
        category.append(row['category'])
        content.append(row['content'])
        
    news_df = pd.DataFrame({'title': title, 'content': content, 'category': category})

    news_df['title'] = news_df['title'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    news_df['title'] = news_df['title'].apply(lambda x: "".join([char for char in x if char not in string.punctuation]))
    news_df['title'] = news_df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    news_df['title'] = news_df['title'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    news_df['title'][:10].apply(lambda x: " ".join([lem.lemmatize(word, 'v') for word in x.split()]))
    news_df['title'] = news_df['title'].apply(lambda x: " ".join([lem.lemmatize(word) for word in x.split()]))
    news_df['title'] = news_df['title'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 3]))
    return news_df


def getCategory(input_value, news_df, count):
    while count < 1:
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
        count += 1
    
    # Additional testing with new data
    extra_test_data = remove_noise(input_value)

    # Convert extra test data to feature vectors
    extra_test_vectors = vectorizer.transform([extra_test_data])
    # Make predictions on extra test data
    extra_prediction = model.predict(extra_test_vectors)

    return str(extra_prediction[0])


def remove_noise(string_input):
    string_df = pd.DataFrame({'title': [string_input]})
    string_df['title'] = string_df['title'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    string_df['title'] = string_df['title'].apply(lambda x: "".join([char for char in x if char not in string.punctuation]))
    string_df['title'] = string_df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    string_df['title'] = string_df['title'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    string_df['title'][:10].apply(lambda x: " ".join([lem.lemmatize(word, 'v') for word in x.split()]))
    string_df['title'] = string_df['title'].apply(lambda x: " ".join([lem.lemmatize(word) for word in x.split()]))
    string_df['title'] = string_df['title'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 3]))
    # to return the string
    string_df = string_df['title'][0]
    return string_df

def recommendNews(categories):
    recommended_news = []
    for category in categories:
        title = []
        categoryList = []
        content = []

        for i in range(1, 6):
                url = f'https://inshorts.com/api/en/search/trending_topics/{category.lower()}?page={i}&type=NEWS_CATEGORY'
                response = requests.get(url)
                data = ujson.loads(response.text)
                for news in data['data']['news_list']:
                    if 'news_obj' in news:
                        title.append(news['news_obj']['title'])
                        categoryList.append(category.capitalize())
                        content.append(news['news_obj']['content'])
        df = pd.DataFrame({'title': title, 'category': category, 'content': content})
           
        filtered_df = df[df['category'] == category]
        news_sample = filtered_df.sample(n=5, random_state=1)  # Randomly sample 5 news items
        recommended_news.extend(news_sample.values.tolist())  # Add the sampled news items to the recommended_news list
    random.shuffle(recommended_news)  # Shuffle the list of recommended news
    return recommended_news[:5]  # Return only the first 5 news items