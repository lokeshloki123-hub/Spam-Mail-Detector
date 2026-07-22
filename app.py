import streamlit as st
import pickle
import numpy as np
# ---------------- Page Configuration ---------------- #

st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.main-title{
    font-size:42px;
    font-weight:700;
    color:#1E88E5;
    text-align:center;
}

.sub-title{
    font-size:18px;
    text-align:center;
    color:#666666;
    margin-bottom:30px;
}

.result-box{
    padding:20px;
    border-radius:12px;
    border:1px solid #DDDDDD;
    margin-top:20px;
}

.footer{
    text-align:center;
    color:gray;
    padding-top:30px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)
# ---------------------------
# Load Model Files
# ---------------------------

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)

    return model, scaler, columns


model, scaler, word_columns = load_model()

def extract_features(email_text):

    words = email_text.lower().split()

    word_count = {}

    for word in words:

        clean_word = "".join(c for c in word if c.isalpha())

        if clean_word:
            word_count[clean_word] = word_count.get(clean_word, 0) + 1

    features = np.zeros(len(word_columns))

    for i, col in enumerate(word_columns):

        if col in word_count:
            features[i] = word_count[col]

    features = scaler.transform([features])

    return features
# ---------------- Sidebar ---------------- #

st.markdown(
"""
<div class='main-title'>
📧 Spam Mail Detector
</div>

<div class='sub-title'>
Detect Spam Emails & SMS using Machine Learning
</div>
""",
unsafe_allow_html=True
)

st.sidebar.title("📌 Project Details")

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("""
### 🤖 Algorithm
Best Performing Model
(Automatically Selected)

Among(Naive bayes,LogisticRegression,SVM)


### 📊 Features
3000 Word Features

### 📂 Dataset
SMS Spam Collection

### 👨‍💻 Developer
Lokesh
""")
# ---------------- Main Page ---------------- #

st.title("📧 Spam Mail Detection")

st.write(
"""
Welcome!

This application predicts whether an Email or SMS is **Spam** or **Not Spam**
using a Machine Learning model.
"""
)

st.markdown("---")

email = st.text_area(
    "✉️ Enter your Email or SMS",
    placeholder="Paste your email or SMS here...",
    height=220
)

predict = st.button("🚀 Predict")

if predict:

    if email.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Feature Extraction
        features = extract_features(email)

        # Prediction
        prediction = model.predict(features)[0]

        # -----------------------------
        # Confidence Score
        # -----------------------------
        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities) * 100

        elif hasattr(model, "decision_function"):
            score = model.decision_function(features)[0]
            confidence = min(99.9, max(50.0, 50 + abs(score) * 10))

        # -----------------------------
        # Suspicious Keywords
        # -----------------------------
        spam_keywords = [
            "free", "win", "winner", "won", "claim",
            "cash", "prize", "urgent", "offer",
            "click", "limited", "money", "reward",
            "gift", "bonus", "selected", "discount",
            "loan", "credit", "buy", "order"
        ]

        words = email.lower().split()

        found_keywords = []

        for word in words:
            clean_word = "".join(c for c in word if c.isalpha())

            if clean_word in spam_keywords:
                found_keywords.append(clean_word)

        # -----------------------------
        # Display Result
        # -----------------------------
        st.markdown("---")

        if prediction == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")

        # Confidence
        if confidence is not None:
            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

        # Keywords
        if len(found_keywords) > 0:
            st.warning("⚠️ Suspicious Keywords Found")

            cols = st.columns(3)

            for i, word in enumerate(found_keywords):
                cols[i % 3].success(word)

        else:
            st.info("No suspicious keywords detected.")
st.markdown("---")

st.markdown(
"""
<div class='footer'>

Made with ❤️ using <b>Streamlit</b>

<br><br>

<b>Spam Mail Detector</b>

<br>

Machine Learning Project

<br><br>

Developed by <b>Lokesh</b>

</div>
""",
unsafe_allow_html=True
)
