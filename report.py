"""
report.py - read the database and show a spending summary.

Run:  python report.py
"""

import pandas as pd

from db import get_connection
import sys

sys.stdout.reconfigure(encoding="utf-8")   # lets Windows consoles print the ₹ symbol

conn = get_connection()

# pd.read_sql_query runs a SQL question against the database and returns a DataFrame
df = pd.read_sql_query("SELECT * FROM transactions", conn)
conn.close()

if df.empty:
    print("No transactions yet. Put CSV files in data/incoming/ and run process_transactions.py")
    raise SystemExit

print(f"Transactions: {len(df)}")
print(f"Total spending: ₹{df['amount'].sum():,.0f}")

print("\nSpending by category:")
print(df.groupby("category")["amount"].sum().sort_values(ascending=False).map("₹{:,.0f}".format).to_string())

# Spending by month: take the first 7 characters of the date ('2026-10')
df["month"] = df["date"].str[:7]
print("\nSpending by month:")
print(df.groupby("month")["amount"].sum().map("₹{:,.0f}".format).to_string())

print("\nTop 3 merchants:")
print(df.groupby("description")["amount"].sum().nlargest(3).map("₹{:,.0f}".format).to_string())

pending = int(df["needs_review"].sum())
if pending:
    print(f"\n{pending} transaction(s) need review -> run: python review.py")
