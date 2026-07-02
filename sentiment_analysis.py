"""
Simple User Feedback Analyzer
=============================
Loads a feedback dataset, trains a Logistic Regression model to classify
feedback as Positive / Negative / Neutral, evaluates it, and lets you test
your own sentences.

Run:  python sentiment_analysis.py
"""

import re
import string

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. LOAD THE DATASET ------------------------------------------------------
df = pd.read_csv("dataset/feedback.csv")   # columns: feedback, sentiment
df = df.dropna()                           # remove empty rows
print("Dataset loaded:", df.shape)
print(df["sentiment"].value_counts(), "\n")


# 2. CLEAN THE TEXT --------------------------------------------------------
def clean_text(text):
    text = str(text).lower()                       # lowercase
    text = re.sub(r"\d+", " ", text)               # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()       # remove extra spaces
    return text

df["clean"] = df["feedback"].apply(clean_text)


# 3. SPLIT INTO FEATURES (X) AND LABEL (y) ---------------------------------
X = df["clean"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. CONVERT TEXT TO NUMBERS (TF-IDF) --------------------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 5. TRAIN THE MODEL -------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


# 6. EVALUATE --------------------------------------------------------------
y_pred = model.predict(X_test_vec)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 7. SAVE THE MODEL --------------------------------------------------------
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel saved to sentiment_model.pkl")


# 8. PREDICT YOUR OWN FEEDBACK ---------------------------------------------
def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()
    return label, round(confidence * 100, 1)

print("\n--- Try it out ---")
for sentence in ["I loved the service.",
                 "The product quality was terrible.",
                 "This is okay."]:
    label, conf = predict(sentence)
    print(f"{sentence}  ->  {label} ({conf}%)")
