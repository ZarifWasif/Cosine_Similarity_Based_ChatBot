import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load data
data_path = "FAQ_QNA.csv"
df = pd.read_csv(data_path, encoding='unicode_escape')
questions_list = df['Questions'].tolist()
answers_list = df['Answers'].tolist()

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

def preprocess_with_stopwords(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

# Fit the vectorizer once
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform([preprocess(q) for q in questions_list])

def get_response(text):
    processed_text = preprocess_with_stopwords(text)
    vectorized_text = vectorizer.transform([processed_text])
    similarities = cosine_similarity(vectorized_text, X)
    max_similarity = np.max(similarities)
    if max_similarity > 0.6:
        high_similarity_questions = [q for q, s in zip(questions_list, similarities[0]) if s > 0.6]
        target_answers = []
        for q in high_similarity_questions:
            q_index = questions_list.index(q)
            target_answers.append(answers_list[q_index])
        Z = vectorizer.transform([preprocess_with_stopwords(q) for q in high_similarity_questions])
        final_similarities = cosine_similarity(vectorized_text, Z)
        closest = np.argmax(final_similarities)
        return target_answers[closest]
    else:
        return "I can't answer this question."
