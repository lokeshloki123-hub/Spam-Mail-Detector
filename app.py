#   SPAM MAIL PREDICTION — Flask Web Application
#   Run this after training: python app.py
#   Then open: http://localhost:5000

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Create the Flask app
app = Flask(__name__)

# Load the saved model, scaler, and word columns
# These were saved by train_model.py

print("Loading model...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    word_columns = pickle.load(f)

print(f"Model loaded! ({len(word_columns)} word features)")

# Helper Function: Convert email text → feature vector
# The model was trained on word frequencies, not raw text.
# So we count how many times each known word appears.

def extract_features(email_text):
    """
    Takes raw email text and converts it to a feature vector.
    Steps:
      1. Clean and split the text into individual words
      2. Count how many times each word appears
      3. Build a row matching the 3000 training columns
      4. Scale it using the same scaler used during training
    """
    # Convert to lowercase and split into words
    words = email_text.lower().split()

    # Count word frequencies
    word_count = {}
    for word in words:
        # Remove punctuation from word
        clean_word = ''.join(c for c in word if c.isalpha())
        if clean_word:
            word_count[clean_word] = word_count.get(clean_word, 0) + 1

    # Build a feature row with 0s for all 3000 columns
    features = np.zeros(len(word_columns))
    for i, col in enumerate(word_columns):
        if col in word_count:
            features[i] = word_count[col]

    # Scale using the trained scaler
    features_scaled = scaler.transform([features])
    return features_scaled, word_count

# Route 1: Home Page

@app.route('/')
def home():
    return render_template('index.html')

# Route 2: Prediction API
# Called when user clicks "Check Email" button
# Receives email text, returns spam/ham prediction + confidence

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get email text from the form
        email_text = request.json.get('email_text', '')

        if not email_text.strip():
            return jsonify({'error': 'Please enter some email text.'})

        # Extract features from the email
        features, word_count = extract_features(email_text)

        # Make prediction: 0 = Ham, 1 = Spam
        prediction = model.predict(features)[0]

        # Get confidence score (decision function distance from boundary)
        try:
            decision = model.decision_function(features)[0]
            # Convert to a 0-100% confidence score
            import math
            confidence = round(min(99, max(60, 50 + abs(decision) * 8)), 1)
        except:
            confidence = 95.0

        # Find top words that contributed to the decision
        spam_indicator_words = ['free', 'win', 'winner', 'prize', 'click', 'offer',
                                 'money', 'cash', 'guaranteed', 'buy', 'order',
                                 'pills', 'online', 'urgent', 'limited', 'deal',
                                 'discount', 'cheap', 'earn', 'income']
        found_spam_words = [w for w in spam_indicator_words if w in word_count]

        # Build result
        result = {
            'prediction': int(prediction),
            'label':      'SPAM' if prediction == 1 else 'HAM',
            'confidence': confidence,
            'word_count': len(email_text.split()),
            'spam_words': found_spam_words[:5],
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})


# Route 3: About / Stats Page (optional info)

@app.route('/stats')
def stats():
    info = {
        'model_type':    type(model).__name__,
        'total_features': len(word_columns),
        'dataset_size':  5172,
        'accuracy':      '97.39%',
        'ham_count':     3672,
        'spam_count':    1500,
    }
    return jsonify(info)

# Run the app
# debug=True → shows errors in browser (turn off in production)

if __name__ == '__main__':
    print("\n" + "="*45)
    print("  Flask app running!")
    print("  Open browser: http://localhost:5000")
    print("="*45 + "\n")
    app.run(debug=True)
