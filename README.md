# 📝 Simple user Feedback Analyzer for website

This web app uses machine learning to classify user feedback as **Positive**, **Negative**, or **Neutral**.

---

## 🔍 Model Used

- **Vectorizer:** TF-IDF (`TfidfVectorizer`)
- **Classifier:** Logistic Regression (`LogisticRegression`)
- **Library:** scikit-learn
- **Target Classes:**
  - `0`: Negative 😡
  - `1`: Positive 😊
  - `2`: Neutral 😐

---

## DATASET

A small sample dataset with 10 feedback sentences. Each sentence is labeled as either:
- Positive (1)
- Negative (0)
- Neutral (2)

Example:

| Feedback               | Label    |
|------------------------|----------|
| "Amazing service"      | Positive |
| "Terrible support"     | Negative |
| "It was okay"          | Neutral  |

---

## 📈 Evaluation

- **Algorithm:** Logistic Regression
- **Vectorization:** TF-IDF
- **Accuracy:** ~60–100% (varies per run)
- **Metrics:** Precision, Recall, F1-score via `classification_report`
- **Confusion Matrix:** Used to verify prediction accuracy per class

---

## 🚀 How to Run

### 1. Clone or Download

```bash
git clone https://github.com/yourusername/feedback-analyzer.git
cd feedback-analyzer
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ✅ Project Structure

```
feedback-analyzer/
├── app.py
├── feedback_model.pkl
├── requirements.txt
├── README.md
└── templates/
    └── feedback.html
```

---

## 📌 Disclaimer

This model is trained on a small dataset for demonstration only. For production use, retrain the model with a larger and more diverse dataset.

for any doubt and clarifications feel free to contact🚀
## contact number - 9871649925
## email - nikita17.mishra@gmail.com
Powered by Civora Nexus and SECT.
