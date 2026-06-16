import streamlit as st
import pandas as pd
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

faq_data = pd.read_csv("AI.csv")

def preprocess_text(text):

    text = text.lower()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]

    return " ".join(tokens)

faq_data["Processed_Question"] = faq_data["Question"].apply(
    preprocess_text
)

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    faq_data["Processed_Question"]
)

def get_response(question):

    processed_question = preprocess_text(question)

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    index = similarity_scores.argmax()

    return faq_data.iloc[index]["Answer"]

st.title("AI FAQ Chatbot")

user_question = st.text_input(
    "Ask a question about AI"
)

if st.button("Ask"):

    response = get_response(user_question)

    st.success(response)