# 📧 Spam Mail Detector using Machine Learning

A professional **Spam Mail Detection** web application built with **Python, Machine Learning, and Streamlit**. This project classifies Email/SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) techniques and supervised machine learning algorithms.

---

## 🚀 Live Demo

🔗 **Live Application:** YOUR_STREAMLIT_APP_URL

---

## 📌 Project Overview

Spam messages are a common cybersecurity and communication problem. This project applies **Natural Language Processing (NLP)** and **Machine Learning** to automatically detect whether a given email or SMS is spam.

The application provides a simple and user-friendly interface where users can paste a message and instantly receive a prediction.

---

## ✨ Features

- 📧 Detects Spam and Non-Spam messages
- 🤖 Machine Learning based prediction
- 📊 Displays prediction confidence
- 🔍 Highlights suspicious spam keywords
- ⚡ Fast and responsive Streamlit interface
- 💻 Easy to deploy and use
- 📱 Clean and professional UI

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn

### NLP
- NLTK

### Web Framework
- Streamlit

### Data Processing
- NumPy
- Pandas

---

## 📂 Project Structure

```
Spam-Mail-Detector/
│
├── app.py                # Streamlit Application
├── train_model.py        # Model Training Script
├── model.pkl             # Trained Machine Learning Model
├── scaler.pkl            # Feature Scaler
├── columns.pkl           # Feature Columns
├── emails.csv            # Dataset
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The model is trained using an Email/SMS spam dataset containing labeled messages.

Target Classes:

- Spam
- Ham (Not Spam)

---

## 🤖 Machine Learning Models Evaluated

The following models were trained and compared:

- ✅ Multinomial Naive Bayes
- ✅ Logistic Regression
- ✅ Support Vector Machine (SVM)

The model with the **highest accuracy** was automatically selected and saved for deployment.

---

## ⚙️ Workflow

1. Load Dataset
2. Data Cleaning
3. Text Preprocessing
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Select Best Model
8. Save Trained Model
9. Deploy using Streamlit

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/lokeshloki123-hub/Spam-Mail-Detector.git
```

### Move to Project Folder

```bash
cd Spam-Mail-Detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 📈 Example Prediction

### Input

```
Congratulations!

You have won ₹50,000.

Click here to claim your free reward.
```

### Output

```
🚨 Spam

Confidence: 98.74%

Suspicious Keywords:
✔ Congratulations
✔ Won
✔ Click
✔ Free
✔ Reward
```

---

## 📸 Application Preview

> Add screenshots after deployment.

Example:

```
screenshots/home.png

screenshots/result.png
```

---

## 📦 Requirements

```
streamlit
scikit-learn
numpy
pandas
nltk
```

---

## 🎯 Future Improvements

- Multiple language support
- Deep Learning models (LSTM/BERT)
- Email attachment analysis
- URL detection
- Real-time Gmail integration
- Dark mode support

---

## 👨‍💻 Author

**Lokesh**

- GitHub: https://github.com/lokeshloki123-hub

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

---

## 📄 License

This project is developed for educational and portfolio purposes.

GitHub:
https://github.com/lokeshloki123-hub

---

