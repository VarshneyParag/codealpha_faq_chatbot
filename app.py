from flask import Flask, request, jsonify, render_template
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# ===== NLTK Setup =====
# Set NLTK data path to a directory where we have write permissions
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True, download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True, download_dir=nltk_data_path)

# ===== FAQ Database =====
faq_pairs = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! Ask me anything about our services.",
    # ... (keep your existing FAQ pairs)
}

# Initialize NLP components after NLTK data is ready
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

# Initialize vectorizer
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
    
    if not query.strip():
        return jsonify({"response": "Please enter a question"})
    
    try:
        cleaned_query = clean_text(query)
        query_vec = vectorizer.transform([cleaned_query])
        similarity = cosine_similarity(query_vec, X)
        best_match = similarity.argmax()
        score = similarity[0][best_match]

        if score < 0.3:
            return jsonify({"response": "Sorry, I couldn't understand that. Could you rephrase?"})
        return jsonify({"response": responses[best_match]})
    except Exception as e:
        return jsonify({"response": "I encountered an error processing your request"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
