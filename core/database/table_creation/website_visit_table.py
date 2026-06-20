#!/usr/bin/env python3
"""
Create the website_visits table in the local SQLite analytics database.
Usage: python website_visit_table.py
"""

import sys

from core.database.operations.visit_tracker import VISITS_DB_PATH, get_connection


def create_website_visits_table():
    """Create the website_visits table if it doesn't already exist."""
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS website_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT NOT NULL,
                user_agent TEXT,
                visit_time TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def main():
    print(f"Creating website_visits table in {VISITS_DB_PATH} ...")
    try:
        create_website_visits_table()
        print("website_visits table ready (id, ip_address, user_agent, visit_time)")
    except Exception as e:
        print(f"Failed to create website_visits table: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
