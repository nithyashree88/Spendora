import joblib
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")   # lets Windows consoles print the ₹ symbol

# ---------- Step 1: load the transactions ----------
file_path = "data/transactions.csv"
transactions = pd.read_csv(file_path)

# ---------- Step 2: let the AI model assign a category ----------
# Load the trained model that train_model.py saved
model = joblib.load("model.joblib")

# predict() takes a list of descriptions and returns one category for each.
# .str.upper() matches how we prepared the text during training.
transactions["category"] = model.predict(transactions["description"].str.upper())

# predict_proba() gives the model's probability for every category.
# .max(axis=1) keeps the highest one = how confident the model is in its answer.
confidence = model.predict_proba(transactions["description"].str.upper()).max(axis=1)
transactions["confidence"] = (confidence * 100).round(0).astype(int).astype(str) + "%"

print("Categorized Transactions:")
print(transactions)

# ---------- Reports ----------
total_spending = transactions["amount"].sum()
print("\nTotal Spending: ₹", total_spending)

print("\nSpending by Category:")
print(transactions.groupby("category")["amount"].sum().sort_values(ascending=False))

highest = transactions.loc[transactions["amount"].idxmax()]
print("\nHighest Transaction:")
print(highest)

print("\nAverage Transaction: ₹", round(transactions["amount"].mean(), 2))

print("\nTransactions above ₹500:")
print(transactions[transactions["amount"] > 500])

# ---------- Try the model on descriptions it has NEVER seen ----------
new_descriptions = ["ZOMATO ORDER 55231", "UBER INDIA TRIP", "APOLLO CLINIC", "HOTSTAR"]
print("\nPredictions for new merchants:")
for text, category in zip(new_descriptions, model.predict([d.upper() for d in new_descriptions])):
    print(f"  {text:<22} -> {category}")
