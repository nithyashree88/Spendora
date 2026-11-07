"""
process_transactions.py - Step 3: automated processing.

Every time it runs it:
  1. Finds every CSV file in data/incoming/
  2. CLEANS the rows (fixes spacing, dates, amounts; drops broken rows)
  3. Gives every transaction a unique ID so duplicates can be detected
  4. Uses the AI model to assign a category + confidence
  5. Saves new rows into the database (skipping ones it has seen before)
  6. Moves the finished file to data/processed/

Run from the terminal:  python process_transactions.py
The web API (api.py) also calls process_incoming() when you upload a file.
"""

import hashlib
import shutil
from pathlib import Path

import joblib
import pandas as pd

from db import get_connection

INCOMING = Path("data/incoming")
PROCESSED = Path("data/processed")
MODEL_PATH = "model.joblib"

# If the model is less sure than this, a human should double-check.
REVIEW_THRESHOLD = 0.60


def clean(df):
    """Fix messy data. Returns (clean_rows, number_of_rows_dropped)."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()   # ' Date ' -> 'date'

    # Descriptions: blank cell -> "", trim spaces, UPPERCASE (same as training)
    df["description"] = df["description"].fillna("").astype(str).str.strip().str.upper()

    # Dates: text -> real dates. errors="coerce": unreadable value -> missing (NaT)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Amounts: remove ₹ and commas ("₹1,299" -> 1299) then convert to a number
    df["amount"] = df["amount"].astype(str).str.replace(r"[₹,\s]", "", regex=True)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date", "amount"])
    df = df[df["description"] != ""]
    df = df[df["amount"] > 0]
    dropped = before - len(df)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")   # store dates as text
    return df.reset_index(drop=True), dropped


def add_transaction_ids(df):
    """
    Fingerprint = date + description + amount (hashed).
    Same transaction -> same fingerprint -> the database refuses the second copy.

    Catch: you can genuinely buy the same thing twice in one day. So identical
    rows INSIDE one file get a counter (0, 1, 2...) added. They stay separate,
    but re-importing the same file produces the same fingerprints, so nothing
    is added twice.
    """
    df = df.copy()
    # 2 decimals so 220, 220.0 and "220.00" all give the SAME text
    amount_text = df["amount"].map(lambda a: f"{a:.2f}")
    key = df["date"] + "|" + df["description"] + "|" + amount_text
    df["occurrence"] = key.groupby(key).cumcount()
    df["transaction_id"] = [
        hashlib.sha256(f"{k}|{n}".encode()).hexdigest()[:16]
        for k, n in zip(key, df["occurrence"])
    ]
    return df


def process_file(path, model, conn):
    """Import one CSV. Returns a dict of counts."""
    result = {"file": path.name, "added": 0, "duplicates": 0, "dropped": 0, "needs_review": 0}
    raw = pd.read_csv(path)
    df, dropped = clean(raw)
    result["dropped"] = dropped
    if df.empty:
        return result

    df = add_transaction_ids(df)

    # AI step: category + confidence
    df["category"] = model.predict(df["description"])
    df["confidence"] = model.predict_proba(df["description"]).max(axis=1).round(2)
    df["needs_review"] = (df["confidence"] < REVIEW_THRESHOLD).astype(int)

    try:
        for row in df.itertuples():
            # INSERT OR IGNORE: insert, but silently skip if transaction_id exists.
            # cursor.rowcount is 1 if inserted, 0 if skipped.
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO transactions
                (transaction_id, date, description, amount, category,
                 confidence, needs_review, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (row.transaction_id, row.date, row.description, float(row.amount),
                 row.category, float(row.confidence), int(row.needs_review), path.name),
            )
            result["added"] += cur.rowcount
            if cur.rowcount:
                result["needs_review"] += int(row.needs_review)
        result["duplicates"] = len(df) - result["added"]
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return result


def process_incoming():
    """Process every CSV in data/incoming/. Returns a list of result dicts."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    INCOMING.mkdir(parents=True, exist_ok=True)

    files = sorted(INCOMING.glob("*.csv"))
    if not files:
        return []

    model = joblib.load(MODEL_PATH)
    conn = get_connection()
    results = []
    for path in files:
        try:
            results.append(process_file(path, model, conn))
        except Exception as error:   # a bad file must not stop the others
            results.append({"file": path.name, "error": str(error)})
            continue
        # Move the file so we don't process it again. Add a number if the name exists.
        target = PROCESSED / path.name
        n = 1
        while target.exists():
            target = PROCESSED / f"{path.stem}_{n}{path.suffix}"
            n += 1
        shutil.move(str(path), str(target))
    conn.close()
    return results


def main():
    results = process_incoming()
    if not results:
        print("No CSV files found in data/incoming/. Nothing to do.")
        return
    for r in results:
        print(f"\nProcessed {r['file']}")
        if "error" in r:
            print(f"  ERROR: {r['error']}  (file left in data/incoming/)")
            continue
        if r["dropped"]:
            print(f"  Dropped {r['dropped']} invalid row(s)")
        print(f"  Added {r['added']} new | Skipped {r['duplicates']} duplicate(s)")

    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM transactions WHERE needs_review = 1").fetchone()[0]
    conn.close()
    print(f"\nDatabase now holds {total} transactions. {pending} need review.")
    if pending:
        print("Run: python review.py   (or use the dashboard)")


if __name__ == "__main__":
    main()
