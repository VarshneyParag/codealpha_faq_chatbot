from flask import Flask, request, jsonify, render_template
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Downloads (first run only)
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# === 🔥 Custom FAQ base: Add more here! ===
faq_pairs = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! Ask me anything about our services.",
    "how are you": "I'm just a bot, but I'm functioning perfectly!",
    "what is your name": "I'm your friendly FAQ Chatbot 🤖",
    "who created you": "I was built by an AI developer for your service.",
    "what can you do": "I can answer FAQs about our services, orders, and more.",
    "how to track my order": "Visit the tracking page and enter your order ID.",
    "return policy": "Returns are accepted within 30 days with original packaging.",
    "refund process": "Refunds are processed within 5-7 business days.",
    "cancel my order": "Please go to your orders and click cancel next to the item.",
    "how long for delivery": "Delivery usually takes 3-5 business days.",
    "do you have a mobile app": "Yes! It's available on both Android and iOS.",
    "how to contact support": "You can reach out at support@example.com",
    "tell me a joke": "Why did the chatbot get promoted? Because it had great 'response' time! 😂",
    "bye": "Goodbye! Have a great day 😊",
    "thank you": "You're welcome!",
    "what about you": "I'm here to help you out 24/7!",
    "do you love me": "I love answering your questions!",
    "can you help me": "Absolutely! What do you need help with?",
    # Add more 80+ as needed
}

# Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

corpus = list(faq_pairs.keys())
responses = list(faq_pairs.values())
cleaned_corpus = [clean_text(q) for q in corpus]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_corpus)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    query = data.get("query", "")
    cleaned_query = clean_text(query)
    query_vec = vectorizer.transform([cleaned_query])
    similarity = cosine_similarity(query_vec, X)
    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score < 0.3:
        return jsonify({"response": "Sorry, I couldn't understand that. Could you rephrase?"})
    return jsonify({"response": responses[best_match]})

if __name__ == "__main__":
    app.run(debug=True)
