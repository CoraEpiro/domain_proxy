#!/usr/bin/env python3
"""Script to fix/update BSR data - remove old entries and set correct values."""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.db"

# Correct BSR data provided by user
trueseamoss_bsr = {
    "2025-11-30": 86,
    "2025-12-01": 84,
    "2025-12-02": 90,
    "2025-12-03": 92,
}

herbalvineyard_bsr = {
    "2025-12-01": 22201,
    "2025-12-02": 22506,
    "2025-12-03": 19050,
}

def fix_bsr_data():
    """Update BSR data to correct values."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        # Delete ALL existing BSR entries to start fresh
        print("Clearing all existing BSR entries...")
        conn.execute("DELETE FROM manual_bsr")
        conn.commit()
        
        # Add correct Trueseamoss BSR data
        print("\nAdding correct Trueseamoss BSR data...")
        for date, bsr in trueseamoss_bsr.items():
            conn.execute(
                "INSERT INTO manual_bsr(date, brand, bsr) VALUES(?, ?, ?)",
                (date, "Trueseamoss", float(bsr))
            )
            print(f"  {date}: {bsr}")
        
        # Add correct HerbalVineyard BSR data
        print("\nAdding correct HerbalVineyard BSR data...")
        for date, bsr in herbalvineyard_bsr.items():
            conn.execute(
                "INSERT INTO manual_bsr(date, brand, bsr) VALUES(?, ?, ?)",
                (date, "HerbalVineyard", float(bsr))
            )
            print(f"  {date}: {bsr}")
        
        conn.commit()
        print("\n✅ All BSR data updated successfully!")
        
        # Verify
        print("\nVerifying database contents:")
        rows = conn.execute("SELECT date, brand, bsr FROM manual_bsr ORDER BY brand, date").fetchall()
        for row in rows:
            print(f"  {row[0]} | {row[1]} | {row[2]}")

if __name__ == "__main__":
    print("Fixing BSR data...")
    fix_bsr_data()
    print("\nDone!")


