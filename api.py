"""
api.py - Step 5: a REST API (+ dashboard) for SmartSpend AI.

An API is a set of web addresses (URLs) that a program - or a web page - can
call to get data or perform actions. FastAPI makes that easy.

Start the server:   python -m uvicorn api:app --reload
Dashboard:          http://127.0.0.1:8000
Interactive docs:   http://127.0.0.1:8000/docs   (try every endpoint in the browser!)
"""

import shutil
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from analytics import get_alerts, load_budgets
from db import get_connection
from process_transactions import INCOMING, MODEL_PATH, process_incoming
from review import CATEGORIES, apply_correction

app = FastAPI(title="SmartSpend AI", version="1.0")


# ------------------------------------------------------------------ the model
_model = None


def get_model():
    """Load the trained model the first time it is needed, then reuse it."""
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run: python train_model.py",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def rows_to_dicts(rows):
    return [dict(r) for r in rows]


# --------------------------------------------------------------- request shapes
class PredictRequest(BaseModel):
    description: str


class CorrectionRequest(BaseModel):
    category: str


# ------------------------------------------------------------------ endpoints
@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the dashboard web page."""
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_trained": Path(MODEL_PATH).exists()}


@app.get("/api/transactions")
def list_transactions(category: str | None = None, limit: int = 50):
    """Newest transactions first. Optional ?category=Food&limit=20"""
    limit = max(1, min(limit, 500))
    conn = get_connection()
    if category:
        rows = conn.execute(
            "SELECT * FROM transactions WHERE category = ? ORDER BY date DESC, rowid DESC LIMIT ?",
            (category, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY date DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return rows_to_dicts(rows)


@app.get("/api/summary")
def summary():
    """Totals by category and by month."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    if df.empty:
        return {"count": 0, "total": 0, "by_category": [], "by_month": []}
    df["month"] = df["date"].str[:7]
    by_category = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    by_month = df.groupby("month")["amount"].sum()
    return {
        "count": int(len(df)),
        "total": float(df["amount"].sum()),
        "by_category": [{"category": k, "amount": float(v)} for k, v in by_category.items()],
        "by_month": [{"month": k, "amount": float(v)} for k, v in by_month.items()],
    }


@app.get("/api/alerts")
def alerts():
    """Unusual spending, recurring payments and the month-end projection."""
    return get_alerts()


@app.get("/api/budgets")
def budgets():
    return load_budgets()


@app.get("/api/review")
def review_queue():
    """Transactions where the AI was not confident enough."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM transactions WHERE needs_review = 1 ORDER BY date DESC"
    ).fetchall()
    conn.close()
    return {"categories": CATEGORIES, "items": rows_to_dicts(rows)}


@app.post("/api/review/{transaction_id}")
def correct(transaction_id: str, body: CorrectionRequest):
    """Save a human-approved category (also teaches the AI at next training)."""
    if body.category not in CATEGORIES:
        raise HTTPException(400, f"Category must be one of: {', '.join(CATEGORIES)}")
    conn = get_connection()
    ok = apply_correction(conn, transaction_id, body.category)
    conn.close()
    if not ok:
        raise HTTPException(404, "Transaction not found")
    return {"status": "saved", "category": body.category}


@app.post("/api/predict")
def predict(body: PredictRequest):
    """Ask the AI: what category is this merchant? Returns the top 3 guesses."""
    text = body.description.strip().upper()
    if not text:
        raise HTTPException(400, "Description cannot be empty")
    model = get_model()
    probabilities = model.predict_proba([text])[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda p: p[1], reverse=True)[:3]
    return {
        "description": text,
        "category": ranked[0][0],
        "confidence": round(float(ranked[0][1]), 2),
        "top3": [{"category": c, "confidence": round(float(p), 2)} for c, p in ranked],
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a bank-statement CSV: it is cleaned, categorized and de-duplicated."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")
    get_model()  # fail early with a clear message if the model is missing

    INCOMING.mkdir(parents=True, exist_ok=True)
    # Path(...).name strips any folder parts, so a malicious filename like
    # '../../x.csv' cannot write outside data/incoming/
    safe_name = Path(file.filename).name
    destination = INCOMING / safe_name
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    results = process_incoming()
    mine = [r for r in results if r["file"] == safe_name]
    if not mine:
        raise HTTPException(500, "File was not processed")
    if "error" in mine[0]:
        # Remove the unreadable file so it is not retried on every future upload
        destination.unlink(missing_ok=True)
        # Show the user a friendly reason, e.g. missing columns
        raise HTTPException(
            400, f"Could not read file. It needs columns: date, description, amount. ({mine[0]['error']})"
        )
    return mine[0]
