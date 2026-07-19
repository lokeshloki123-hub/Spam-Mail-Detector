
# Step 1: Import the tools we need
import pandas as pd          
import numpy as np           
import pickle                

from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import MinMaxScaler         
from sklearn.naive_bayes import MultinomialNB          
from sklearn.linear_model import LogisticRegression   
from sklearn.svm import LinearSVC                      
from sklearn.metrics import accuracy_score, classification_report

print("=" * 55)
print("   SPAM MAIL PREDICTION — MODEL TRAINING")
print("=" * 55)


# Step 2: Load the Dataset

print("\n[1/5] Loading dataset...")
df = pd.read_csv('emails.csv')
print(f"      Loaded {df.shape[0]} emails with {df.shape[1]} columns")


# Step 3: Prepare Features (X) and Labels (y)

print("\n[2/5] Preparing data...")

# Get all word columns (skip 'Email No.' and 'Prediction')
word_columns = [col for col in df.columns if col not in ['Email No.', 'Prediction']]

X = df[word_columns]         
y = df['Prediction']         

print(f"      Features (word columns): {len(word_columns)}")
print(f"      Ham emails : {(y == 0).sum()}")
print(f"      Spam emails: {(y == 1).sum()}")

# Step 4: Scale the features using MinMaxScaler

print("\n[3/5] Scaling features (MinMaxScaler)...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("      All values scaled to range [0, 1] ✅")

# Step 5: Split into Training and Testing sets

print("\n[4/5] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      
    random_state=42,    
    stratify=y          
)
print(f"      Training samples : {len(X_train)}")
print(f"      Testing samples  : {len(X_test)}")

# Step 6: Train 3 Models and Compare

print("\n[5/5] Training models...")

# Model 1: Naive Bayes — works on probability
#   "If this email contains 'free', 'win', 'prize' → probably spam"
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
print(f"      Naive Bayes        → Accuracy: {nb_acc*100:.2f}%")

# Model 2: Logistic Regression — draws a line between spam and ham
#   Uses math to find the best boundary between the two classes

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
print(f"      Logistic Regression→ Accuracy: {lr_acc*100:.2f}%")

# Model 3: SVM (Support Vector Machine) — finds the widest margin
#   Tries to separate spam and ham with maximum gap between them

svm_model = LinearSVC(max_iter=2000, random_state=42)
svm_model.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
print(f"      SVM                → Accuracy: {svm_acc*100:.2f}%")

# Pick the best model
best_acc   = max(nb_acc, lr_acc, svm_acc)
best_model = {nb_acc: nb_model, lr_acc: lr_model, svm_acc: svm_model}[best_acc]
best_name  = {nb_acc: 'Naive Bayes', lr_acc: 'Logistic Regression',
              svm_acc: 'SVM'}[best_acc]

print(f"\n      Best Model: {best_name} ({best_acc*100:.2f}%) ✅")

# Detailed report
print("\n" + "─"*55)
print(f"Detailed Report ({best_name}):")
print(classification_report(y_test, best_model.predict(X_test),
                              target_names=['Ham (Normal)', 'Spam']))

# Step 7: Save the Model and Scaler to files
# We save using 'pickle' — it's like packaging the model
# into a file so Flask can load and use it later

print("─"*55)
print("Saving model and scaler to files...")

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(word_columns, f)

print("  model.pkl   → Trained ML model saved ✅")
print("  scaler.pkl  → Scaler saved ✅")
print("  columns.pkl → Word column names saved ✅")
print("\nTraining complete! Now run: python app.py")
print("=" * 55)
