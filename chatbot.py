import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = "intents.json"  # Ensure the correct path
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags, patterns = [], []
responses_dict = {}
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
    responses_dict[intent['tag']] = intent['responses']

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    return random.choice(responses_dict.get(tag, ["I'm not sure how to respond to that."]))

# Streamlit UI Enhancements
st.set_page_config(page_title="NLP Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    .title-text {
        font-size: 36px;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
    }
    .chat-container {
        background-color: white;
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with styling
st.sidebar.image("image.png", width=120)  # Reduced size
st.sidebar.title("Chatbot Menu")
menu = ["Home", "Conversation History", "About"]
choice = st.sidebar.radio("Navigation", menu)

# Home Page
if choice == "Home":
    st.markdown("<p class='title-text'>Welcome to the NLP Chatbot 🤖</p>", unsafe_allow_html=True)
    st.write("Talk to the chatbot below!")
    
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv.writer(csvfile).writerow(['User Input', 'Chatbot Response', 'Timestamp'])
    
    user_input = st.text_input("You:", "", key="user_input")
    if user_input:
        response = chatbot(user_input)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
            csv.writer(csvfile).writerow([user_input, response, timestamp])
        
        st.markdown(f"""
        <div class='chat-container'>
            <b>You:</b> {user_input}<br>
            <b>Chatbot:</b> {response}
        </div>
        """, unsafe_allow_html=True)
        
    if st.button("Clear Chat"):
        open('chat_log.csv', 'w').close()
        st.experimental_rerun()

# Conversation History
elif choice == "Conversation History":
    st.header("Conversation History")
    if os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.markdown(f"""
                <div class='chat-container'>
                    <b>User:</b> {row[0]}<br>
                    <b>Chatbot:</b> {row[1]}<br>
                    <i>{row[2]}</i>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.write("No chat history found.")

# About Page
elif choice == "About":
    st.subheader("About This Chatbot")
    st.write("""
    This chatbot is designed to process and respond to user queries using Natural Language Processing (NLP) and Machine Learning techniques.

    - **Natural Language Processing (NLP):** 
      - Tokenizes and processes user input to extract meaningful patterns.
      - Converts text into numerical form using TF-IDF (Term Frequency-Inverse Document Frequency) for further analysis.

    - **Machine Learning (Logistic Regression):** 
      - Trained on predefined patterns from intents.json to classify user input.
      - Uses supervised learning to predict the most relevant intent for each user query.

    - **TF-IDF Vectorization:** 
      - Converts textual data into numerical vectors, making it easier for the model to interpret and classify.

    - **Streamlit for UI:** 
      - Provides an interactive web interface for users.
      - Displays chatbot responses in a structured format.
      - Enables conversation history tracking and chat clearing functionality.

    - **Data Logging (CSV-based):** 
      - Stores chat history in a CSV file for later review.
      - Helps improve model performance by analyzing user interactions.

    """)
