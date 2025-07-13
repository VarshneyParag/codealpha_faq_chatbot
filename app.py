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
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
required_nltk = ['punkt', 'stopwords']
for resource in required_nltk:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True, download_dir=nltk_data_path)

# ===== FAQ Database =====
faq_pairs = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! Ask me anything about our services.",
    # ... (your existing FAQ pairs)
}

# Initialize NLP components
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

# ... (rest of your Flask routes remain the same)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
