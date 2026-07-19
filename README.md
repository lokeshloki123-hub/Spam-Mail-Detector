# 📧 Spam Mail Detector using Machine Learning & NLP

## 📌 Project Overview

Spam Mail Detector is a Machine Learning-based web application that automatically classifies emails as **Spam** or **Ham (Not Spam)**.

The project uses **Natural Language Processing (NLP)** techniques to analyze email text, extract meaningful features, and predict whether an email is unwanted spam or a legitimate message.

A trained Machine Learning model is integrated with a **Flask web application** to provide real-time spam email predictions.

---

## 🚀 Features

* ✅ Classifies emails into Spam and Ham categories
* ✅ Text preprocessing using NLP techniques
* ✅ Converts email text into numerical features using TF-IDF
* ✅ Machine Learning-based prediction system
* ✅ Flask web interface for real-time prediction
* ✅ Lightweight and easy-to-use application
* ✅ Ready for deployment

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* NumPy
* Pandas
* Scikit-learn
* Flask
* Gunicorn

### Machine Learning & NLP

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Supervised Machine Learning Classification

---

## 🧠 Machine Learning Workflow

```
Data Collection
        |
        ↓
Data Cleaning & Preprocessing
        |
        ↓
Text Feature Extraction (TF-IDF)
        |
        ↓
Model Training
        |
        ↓
Model Evaluation
        |
        ↓
Model Deployment using Flask
```

---

## 🤖 Machine Learning Model

The model follows a supervised learning approach.

### Steps:

1. Email text data is collected
2. Text data is cleaned and preprocessed
3. TF-IDF converts text into numerical vectors
4. Classification model learns patterns from training data
5. The trained model predicts new email categories

### Algorithms Used:

* Multinomial Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)

The best-performing model is saved and used for prediction.

---

## 📂 Project Structure

```
Spam-Mail-Detector/
│
├── app.py                  # Flask application
├── spam_detector.ipynb     # Model training notebook
├── model.pkl               # Trained ML model
├── vectorizer.pkl          # TF-IDF vectorizer
├── requirements.txt        # Required dependencies
├── templates/
│   └── index.html          # Frontend HTML page
├── static/
│   └── style.css           # CSS styling
│
└── README.md               # Project documentation
```

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/lokeshloki123-hub/Spam-Mail-Detector.git
```

### 2. Navigate to project directory

```bash
cd Spam-Mail-Detector
```

### 3. Create virtual environment

```bash
python -m venv venv
```

### 4. Activate virtual environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000/
```

Enter an email message and click **Predict** to check whether it is Spam or Ham.

---

## 📊 Model Evaluation

The model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 📸 Application Preview

(Add screenshots of your Flask application here)

Example:

```
![Spam Detector UI](static/screenshot.png)
```

---

## 🔮 Future Improvements

* Improve accuracy using Deep Learning models
* Implement BERT/LSTM-based email classification
* Add email attachment scanning
* Integrate with real email services
* Deploy using cloud platforms

---

## 👨‍💻 Author

**Lokesh N**

GitHub:
https://github.com/lokeshloki123-hub

---

