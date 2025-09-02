from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

# --- Download NLTK stopwords if not already present ---
try:
    stopwords.words('english')
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    print("Download complete.")

# --- Initialize Flask App ---
app = Flask(__name__)

# --- MODIFICATION 1: Define the text cleaning function directly in the app ---
# This ensures the function is always available and resolves the pickle error.
stop_words = set(stopwords.words('english'))
def clean_text_function(text):
    """Cleans input text by removing punctuation, converting to lowercase, and removing stopwords."""
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    return ' '.join([word for word in text.split() if word not in stop_words])


# --- Load Model, Tokenizer, and Label Mapping ---
MODEL_PATH = os.path.join("saved_models", "model")
TOKENIZER_PATH = os.path.join("saved_models", "tokenizer")
PREPROCESSING_INFO_PATH = os.path.join("saved_models", "preprocessing_info.pkl")

# Check if model and tokenizer exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError("The 'saved_models/model' or 'saved_models/tokenizer' directory was not found. Please ensure the model is saved correctly.")

print("Loading model and tokenizer...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print("Model and tokenizer loaded successfully.")

# Load ONLY the label mapping from the pickle file
try:
    with open(PREPROCESSING_INFO_PATH, "rb") as f:
        preprocessing_info = pickle.load(f)
    label_mapping = preprocessing_info['label_mapping']
    print("Preprocessing info loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"The file '{PREPROCESSING_INFO_PATH}' was not found.")
except (KeyError, AttributeError):
     # Fallback in case the pickle file still contains the old structure or is corrupted
    print("Warning: Could not load label_mapping from pickle. Using default: {0: 'fake', 1: 'real'}")
    label_mapping = {0: 'fake', 1: 'real'}


# --- Define Routes ---
@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if request.method == 'POST':
        news_text = request.form['news_text']

        if not news_text.strip():
            return render_template('index.html', prediction_text='Please enter some news text.', original_text=news_text)

        # 1. Preprocess the text using the function defined in this file
        cleaned_text = clean_text_function(news_text)

        # 2. Tokenize the text
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        # 3. Get prediction from model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        prediction = label_mapping.get(predicted_class_idx, "Unknown")

        result_text = f"The news is likely: {prediction.upper()}"
        return render_template('index.html', prediction_text=result_text, original_text=news_text)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)


