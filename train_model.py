"""
train_model.py  -  Step 2: teach the computer to categorize expenses.

Run this ONCE (and again whenever you add rows to data/training_data.csv).
It reads labelled examples, trains a model, tests it, and saves it to model.joblib.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 1. Load the labelled examples (description -> category)
data = pd.read_csv("data/training_data.csv")
print(f"Loaded {len(data)} training examples")
print("Examples per category:")
print(data["category"].value_counts())

# 2. Split into inputs (X) and the answers we want to predict (y)
X = data["description"].str.upper()   # the text, made UPPERCASE so 'swiggy' == 'SWIGGY'
y = data["category"]

# 3. Hold back 20% of the data as an "exam" the model has never seen.
#    random_state=42 makes the split the same every time (repeatable results).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Build the model as a 2-stage pipeline:
#    Stage A (TfidfVectorizer): turn text into numbers.
#       analyzer="char_wb", ngram_range=(2, 4) means it looks at small chunks of
#       letters (e.g. 'SWI', 'WIG', 'IGG'). So an unseen name like 'SWIGGY ORDER 8841'
#       still shares chunks with 'SWIGGY' and gets recognised.
#    Stage B (LogisticRegression): learns which number patterns mean which category.
model = Pipeline([
    ("text_to_numbers", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
    ("classifier", LogisticRegression(C=20, max_iter=1000)),
])

# 5. TRAIN: the model studies the training examples
model.fit(X_train, y_train)

# 6. TEST: ask it to predict the exam questions and compare with the real answers
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy on unseen test data: {accuracy:.0%}")

# 7. Retrain on ALL the data so the final model learns from every example,
#    then save it to a file so main.py can reuse it without retraining.
model.fit(X, y)
joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")
