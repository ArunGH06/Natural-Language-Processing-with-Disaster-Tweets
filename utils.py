import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
def load_dataset(mode):
    if mode == 'train':
        try :
            train_df = pd.read_csv('data/train.csv')
            return train_df
        except :
            print('Error loading the dataset.')
            return None
    elif mode == 'test':
        try :
            test_df = pd.read_csv('data/test.csv')
            return test_df
        except :
            print('Error loading the dataset.')
            return None
    else :
        print('Invalid mode. Please choose either "train" or "test".')
        return None


# Define the cleaning function
def clean_text(text):
    
    # Lowercase the text
    text = text.lower()
        
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back to string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Featurizing text data
def featurize_text(df):

    # Train-test split 
    X = df['text']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorization
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    
    X_test = tfidf.transform(X_test)
    
    return X_train, y_train, X_test, y_test


# Training
def train_model(X,y):
    
    # Initialize the model
    classifier = LogisticRegression()
    classifier.fit(X, y)

    return classifier

# Evaluation
def model_evaluation(model, X_test,y_test):
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
