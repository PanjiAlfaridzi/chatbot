from flask import Flask, request, jsonify, render_template
import nltk
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')
nltk.download('popular', quiet=True)
nltk.download('punkt')  # first-time use only
nltk.download('wordnet')  # first-time use only

# Your chatbot logic (reused from previous chatbot script)
def LemTokens(tokens):
    lemmer = WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Additional chatbot responses
ADDITIONAL_RESPONSES = {
    "how are you": "I'm just a chatbot, but thanks for asking!",
    # Add more responses as needed
}

def response(user_response):
    with open('chatbot.txt', 'r', encoding='utf8') as fin:
        raw = fin.read().lower()
    sent_tokens = nltk.sent_tokenize(raw)
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry, I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    if user_response in ADDITIONAL_RESPONSES:
        robo_response = ADDITIONAL_RESPONSES[user_response]
    sent_tokens.remove(user_response)
    return robo_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('chat.html')  # Melayani halaman utama

@app.route('/chat')
def chat_page():
    return render_template('chat.html')  # Melayani halaman chat

@app.route('/chat', methods=['POST'])
def get_chat_response():
    user_message = request.json['message']
    bot_reply = response(user_message)
    return jsonify({'reply': bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
