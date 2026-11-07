"""
review.py - a human checks the AI's uncertain guesses.

For every transaction flagged needs_review you can:
  - press Enter to accept the AI's category, or
  - type the correct category, or
  - type 's' to skip it for now.

Corrections are also appended to data/training_data.csv. Next time you run
train_model.py the AI learns from YOUR corrections and improves.
(predict -> human corrects -> retrain: how real ML systems get better.)

Run:  python review.py
"""

import csv
import sys

from db import get_connection

TRAINING_FILE = "data/training_data.csv"
CATEGORIES = ["Food", "Groceries", "Shopping", "Travel", "Entertainment", "Bills", "Health"]


def add_to_training_data(description, category):
    with open(TRAINING_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([description, category])


def apply_correction(conn, transaction_id, category):
    """Save a human-approved category. Used by this script AND by the web API."""
    row = conn.execute(
        "SELECT description FROM transactions WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    if row is None:
        return False
    conn.execute(
        "UPDATE transactions SET category = ?, confidence = 1.0, needs_review = 0 "
        "WHERE transaction_id = ?",
        (category, transaction_id),
    )
    conn.commit()
    add_to_training_data(row["description"], category)
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # lets Windows consoles print the ₹ symbol
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM transactions WHERE needs_review = 1 ORDER BY date"
    ).fetchall()

    if not rows:
        print("Nothing to review. All predictions were confident.")
        return

    print(f"{len(rows)} transaction(s) need review.")
    print("Categories:", ", ".join(CATEGORIES))
    fixed = 0

    for row in rows:
        print(f"\n{row['date']}  {row['description']}  ₹{row['amount']:.0f}")
        print(f"  AI guess: {row['category']} ({row['confidence']:.0%} sure)")
        answer = input("  Enter = accept, or type category, or 's' = skip: ").strip()

        if answer.lower() == "s":
            continue

        category = row["category"] if answer == "" else answer.title()
        if category not in CATEGORIES:
            print(f"  '{answer}' is not a known category - skipped.")
            continue

        apply_correction(conn, row["transaction_id"], category)
        fixed += 1

    conn.close()
    print(f"\nReviewed {fixed} transaction(s). Re-run train_model.py to teach the AI.")


if __name__ == "__main__":
    main()
