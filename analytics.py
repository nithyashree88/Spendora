"""
analytics.py - Step 4: intelligent alerts and predictions.

Three features, each a plain function that takes a DataFrame of transactions:
  1. find_unusual()     - spending that is far above what is normal for that merchant
  2. find_recurring()   - subscriptions / regular bills (Netflix, rent, ...)
  3. project_month()    - will you overspend this month?

Run from the terminal:  python analytics.py
The web API (api.py) calls get_alerts() too.
"""

import calendar
import json
import sys
from pathlib import Path

import pandas as pd

from db import get_connection

BUDGET_FILE = Path("budgets.json")


# ---------------------------------------------------------------- loading
def load_transactions():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_budgets():
    if BUDGET_FILE.exists():
        return json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
    return {"monthly_total": None, "categories": {}}


# ---------------------------------------------------------- 1. unusual spend
def find_unusual(df, factor=2.5, min_history=3):
    """
    Flag a transaction when it is more than `factor` x the MEDIAN amount that
    you normally spend at that merchant.

    Why the median and not the average? One huge purchase drags the average up
    and hides itself. The median (the middle value) ignores extreme values.
    We need at least `min_history` purchases before we trust what 'normal' is.
    """
    unusual = []
    for merchant, group in df.groupby("description"):
        if len(group) < min_history:
            continue
        typical = group["amount"].median()
        for row in group.itertuples():
            if row.amount > factor * typical:
                unusual.append({
                    "transaction_id": row.transaction_id,
                    "date": row.date.strftime("%Y-%m-%d"),
                    "description": merchant,
                    "amount": float(row.amount),
                    "typical_amount": float(typical),
                    "times_normal": round(row.amount / typical, 1),
                    "message": f"{merchant}: ₹{row.amount:,.0f} is {row.amount / typical:.1f}x "
                               f"your usual ₹{typical:,.0f}",
                })
    return sorted(unusual, key=lambda u: u["date"], reverse=True)


# ------------------------------------------------------------- 2. recurring
def find_recurring(df, min_occurrences=3):
    """
    A payment is 'recurring' when the same merchant is paid
      - at least `min_occurrences` times, on different days,
      - at a steady rhythm (about weekly or about monthly), and
      - for nearly the same amount each time (within 10%).
    Variable spending like Swiggy fails the 'same amount' test, so it is ignored.
    """
    recurring = []
    for merchant, group in df.groupby("description"):
        dates = sorted(group["date"].dt.normalize().unique())
        if len(dates) < min_occurrences:
            continue

        # Gaps (in days) between consecutive payments
        gaps = pd.Series(dates).diff().dropna().dt.days
        median_gap = gaps.median()

        amounts = group["amount"]
        spread = (amounts.max() - amounts.min()) / amounts.median()
        if spread > 0.10:
            continue   # amounts vary too much to be a fixed bill

        if 6 <= median_gap <= 8:
            frequency, monthly_factor = "weekly", 52 / 12
        elif 26 <= median_gap <= 35:
            frequency, monthly_factor = "monthly", 1
        else:
            continue

        typical = float(amounts.median())
        last = pd.Timestamp(dates[-1])
        next_due = last + pd.Timedelta(days=int(round(median_gap)))
        recurring.append({
            "description": merchant,
            "frequency": frequency,
            "amount": typical,
            "last_paid": last.strftime("%Y-%m-%d"),
            "next_expected": next_due.strftime("%Y-%m-%d"),
            "monthly_cost": round(typical * monthly_factor, 2),
        })
    return sorted(recurring, key=lambda r: r["monthly_cost"], reverse=True)


# -------------------------------------------------------------- 3. projection
def project_month(df, recurring, budgets, unusual_ids=(), as_of=None, window_days=60):
    """
    Predict this month's total spending.

    Naive idea: (spent so far / days passed) x days in month. That is WRONG when
    rent is paid on the 1st - it would look like you spend ₹15,000 every day!
    Early in the month it is also very noisy (one big order in 5 days ruins it).

    Better idea (what we do):
        projected = spent so far                              (a fact)
                  + recurring bills still due this month      (known, fixed)
                  + typical daily spend x days left           (an estimate)

    'Typical daily spend' = everyday (non-recurring) spending over the last
    `window_days` days, ignoring one-off unusual purchases. Those already
    happened and are inside 'spent so far'; we just don't assume they repeat.
    """
    as_of = as_of or df["date"].max()
    month_start = as_of.replace(day=1).normalize()
    days_in_month = calendar.monthrange(as_of.year, as_of.month)[1]
    month_end = month_start + pd.Timedelta(days=days_in_month - 1)
    days_elapsed = as_of.day
    days_left = days_in_month - days_elapsed

    this_month = df[(df["date"] >= month_start) & (df["date"] <= as_of)]
    spent = float(this_month["amount"].sum())

    recurring_names = {r["description"] for r in recurring}
    # Typical daily everyday-spending over the recent window
    window_start = max(as_of - pd.Timedelta(days=window_days - 1), df["date"].min()).normalize()
    window = df[(df["date"] >= window_start) & (df["date"] <= as_of)]
    everyday = window[
        ~window["description"].isin(recurring_names)
        & ~window["transaction_id"].isin(set(unusual_ids))
    ]
    window_len = (as_of.normalize() - window_start).days + 1
    daily_variable = float(everyday["amount"].sum()) / window_len

    # Recurring payments whose next due date falls later this month
    # (and which have not already been paid this month)
    paid_this_month = set(this_month["description"])
    upcoming = [
        r for r in recurring
        if r["description"] not in paid_this_month
        and as_of < pd.Timestamp(r["next_expected"]) <= month_end
    ]
    upcoming_total = sum(r["amount"] for r in upcoming)

    projected = spent + upcoming_total + daily_variable * days_left
    budget = budgets.get("monthly_total")

    result = {
        "month": as_of.strftime("%B %Y"),
        "as_of": as_of.strftime("%Y-%m-%d"),
        "days_elapsed": days_elapsed,
        "days_left": days_left,
        "spent_so_far": round(spent, 2),
        "upcoming_recurring": [
            {"description": r["description"], "amount": r["amount"], "due": r["next_expected"]}
            for r in upcoming
        ],
        "typical_daily_spend": round(daily_variable, 2),
        "projected_total": round(projected, 2),
        "budget": budget,
    }
    if budget:
        result["over_budget"] = projected > budget
        result["difference"] = round(projected - budget, 2)
        result["percent_of_budget"] = round(projected / budget * 100, 1)

    # Category budgets: how much of each is already used?
    category_status = []
    for category, limit in budgets.get("categories", {}).items():
        used = float(this_month[this_month["category"] == category]["amount"].sum())
        category_status.append({
            "category": category,
            "spent": round(used, 2),
            "budget": limit,
            "percent_used": round(used / limit * 100, 1),
        })
    result["category_budgets"] = category_status
    return result


# ------------------------------------------------------------------ combined
def get_alerts():
    df = load_transactions()
    if df.empty:
        return {"unusual": [], "recurring": [], "projection": None}
    recurring = find_recurring(df)
    unusual = find_unusual(df)
    unusual_ids = [u["transaction_id"] for u in unusual]
    return {
        "unusual": unusual,
        "recurring": recurring,
        "projection": project_month(df, recurring, load_budgets(), unusual_ids),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    alerts = get_alerts()

    print("=== UNUSUAL SPENDING ===")
    for u in alerts["unusual"] or []:
        print(" !", u["date"], u["message"])
    if not alerts["unusual"]:
        print(" None found.")

    print("\n=== RECURRING PAYMENTS ===")
    for r in alerts["recurring"]:
        print(f" - {r['description']:<14} ₹{r['amount']:>8,.0f} {r['frequency']:<8} "
              f"next ~{r['next_expected']}  (≈ ₹{r['monthly_cost']:,.0f}/month)")
    total = sum(r["monthly_cost"] for r in alerts["recurring"])
    print(f"   Fixed commitments: ≈ ₹{total:,.0f} per month")

    p = alerts["projection"]
    if p:
        print(f"\n=== {p['month'].upper()} PROJECTION (data up to {p['as_of']}) ===")
        print(f" Spent so far:       ₹{p['spent_so_far']:,.0f}")
        for u in p["upcoming_recurring"]:
            print(f" Still due:          {u['description']} ₹{u['amount']:,.0f} on {u['due']}")
        print(f" Typical daily spend: ₹{p['typical_daily_spend']:,.0f} x {p['days_left']} days left")
        print(f" Projected total:    ₹{p['projected_total']:,.0f}")
        if p["budget"]:
            verdict = "OVER" if p["over_budget"] else "within"
            print(f" Budget:             ₹{p['budget']:,.0f}  -> {verdict} budget "
                  f"({p['percent_of_budget']}%)")
        print("\n Category budgets used so far:")
        for c in p["category_budgets"]:
            flag = "  <-- over!" if c["percent_used"] > 100 else ""
            print(f"  {c['category']:<14} ₹{c['spent']:>7,.0f} / ₹{c['budget']:,.0f} "
                  f"({c['percent_used']}%){flag}")
