# User Feedback Analyzer

A simple Machine Learning project that classifies user feedback as
**Positive**, **Negative**, or **Neutral** using TF-IDF + Logistic Regression.

## Files

| File | What it is |
|------|------------|
| `sentiment_analysis.py` | The whole project — load data, train, evaluate, predict |
| `dataset/feedback.csv` | The labelled feedback data (6,000 rows) |
| `dataset/generate_dataset.py` | Script that created the sample data (optional) |
| `sentiment_model.pkl` | The trained model (created when you run the script) |
| `vectorizer.pkl` | The saved TF-IDF vectorizer |

## How to run

```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

## What it does

1. Loads `dataset/feedback.csv`
2. Cleans the text (lowercase, remove numbers/punctuation/extra spaces)
3. Splits into 80% training / 20% testing
4. Turns text into numbers with TF-IDF
5. Trains a Logistic Regression model
6. Prints accuracy, classification report, and confusion matrix
7. Saves the model, then predicts a few example sentences

## Example output

```
Accuracy: 0.94

I loved the service.               ->  Positive (90.6%)
The product quality was terrible.  ->  Negative (65.6%)
This is okay.                      ->  Neutral (89.7%)
```

Expected accuracy: **~90–94%**.
