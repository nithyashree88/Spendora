# SmartSpend AI

A personal expense tracker that reads bank statements, categorizes spending with
machine learning, spots unusual spending and subscriptions, predicts whether you will
overspend this month, and shows everything on a web dashboard.

## Folder layout

```
SmartSpendAI/
├── data/
│   ├── incoming/            <- drop bank-statement CSVs here
│   ├── processed/           <- imported files are moved here
│   ├── training_data.csv    <- examples the AI learns from
│   └── transactions.csv     <- tiny file used by main.py (Step 1-2 demo)
├── sample_files/            <- a messy statement for practising
├── static/index.html        <- the dashboard page
├── budgets.json             <- your monthly budgets (edit me!)
├── main.py                  <- Step 1-2 demo: analyze + categorize one CSV
├── train_model.py           <- Step 2: train the AI
├── db.py                    <- Step 3: database helper
├── process_transactions.py  <- Step 3: clean, categorize, de-duplicate, save
├── review.py                <- Step 3: you fix the AI's uncertain guesses
├── report.py                <- Step 3: text summary
├── analytics.py             <- Step 4: unusual spending, recurring, projection
├── api.py                   <- Step 5: REST API + dashboard server
└── requirements.txt
```

## Setup (Windows, VS Code terminal)

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activate script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Run it, in order

| # | Command | What happens |
|---|---------|--------------|
| 1 | `python main.py` | Step 1-2 demo (needs step 2 below first) |
| 2 | `python train_model.py` | Trains the AI, saves `model.joblib` |
| 3 | `python process_transactions.py` | Imports `data/incoming/*.csv` into `smartspend.db` |
| 4 | `python analytics.py` | Prints unusual spending, subscriptions, projection |
| 5 | `python -m uvicorn api:app --reload` | Starts the dashboard |

Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/docs for the API.
Press `Ctrl+C` in the terminal to stop the server.

Run `python train_model.py` BEFORE `main.py`, `process_transactions.py` or the dashboard.

## Practising

* Drop `sample_files/bank_statement_2.csv` into `data/incoming/` and run step 3 again
  (or upload it in the dashboard). It contains bad rows, a duplicate and an unknown merchant.
* Fix the unknown merchant in the dashboard, then run `python train_model.py` again.
* Edit `budgets.json` and refresh the dashboard.
* Add 20+ rows to `data/training_data.csv` from your own spending and retrain.

## Start fresh

Stop the server, then delete `smartspend.db` and `model.joblib`. Put your CSVs
back in `data/incoming/` and repeat steps 2-3.

## Using your own bank data

CSV with columns `date, description, amount` (spending as positive numbers).
If your bank's export has other column names, rename them in the first row.

## Troubleshooting

* **`can't open file '...train_model.py': No such file`** - your terminal is in the wrong
  folder. Windows "Extract All" often creates a folder inside a folder. Run `dir`; you must
  see `train_model.py` and `main.py`. If not, `cd` into the inner `SmartSpendAI` folder.
  In VS Code use File > Open Folder on that inner folder.
* **Dashboard shows no data** - you have not imported yet. Run, in order:
  `python train_model.py` then `python process_transactions.py`, and refresh.
  If it says "No CSV files found", put a CSV in `data/incoming/`.
* **`ModuleNotFoundError`** - the virtual environment is not active or libraries are not
  installed. You should see `(.venv)` in the prompt; then `python -m pip install -r requirements.txt`.
* **`Model not trained yet`** - run `python train_model.py`.
* **Port already in use** - another server is running. Press Ctrl+C in that terminal, or use
  `python -m uvicorn api:app --port 8001`.
