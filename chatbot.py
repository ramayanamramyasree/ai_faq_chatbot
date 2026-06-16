import pandas as pd
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLP resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
faq_data = pd.read_csv("faq_dataset.csv")

# Text preprocessing
def preprocess_text(text):

    text = str(text).lower()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]

    return " ".join(tokens)

# Apply preprocessing
faq_data["Processed_Question"] = faq_data["Question"].apply(
    preprocess_text
)

# Convert questions to vectors
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    faq_data["Processed_Question"]
)

# Response function
def get_response(user_question):

    processed_question = preprocess_text(user_question)

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.2:
        return "Sorry, I couldn't understand that question."

    return faq_data.iloc[best_match_index]["Answer"]

# Chatbot loop
print("="*40)
print(" AI FAQ CHATBOT ")
print("="*40)
print("Type 'exit' to quit")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    answer = get_response(user_input)

    print("Chatbot:", answer)