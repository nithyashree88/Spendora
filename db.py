"""
db.py - the database helper used by every other script.

We use SQLite: a real database stored in ONE file (smartspend.db).
It comes built into Python, so there is nothing extra to install.
"""

import sqlite3

DB_PATH = "smartspend.db"


def get_connection():
    """Open the database (the file is created automatically if it doesn't exist)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us read columns by name: row["amount"]

    # A table is like a spreadsheet sheet. PRIMARY KEY means: no two rows may
    # share the same transaction_id. That rule is what blocks duplicates.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            date           TEXT NOT NULL,
            description    TEXT NOT NULL,
            amount         REAL NOT NULL,
            category       TEXT NOT NULL,
            confidence     REAL NOT NULL,
            needs_review   INTEGER NOT NULL DEFAULT 0,
            source_file    TEXT
        )
        """
    )
    conn.commit()
    return conn
