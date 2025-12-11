#!/usr/bin/env python3
"""Script to add BSR data for both brands."""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.db"

# BSR data provided by user
trueseamoss_bsr = {
    "2025-12-06": 180,
    "2025-12-07": 196,
    "2025-12-08": 204,
    "2025-12-09": 221,
    "2025-12-10": 260,
}

herbalvineyard_bsr = {
    "2025-12-06": 26098,
    "2025-12-07": 24919,
    "2025-12-08": 23382,
    "2025-12-09": 21811,
    "2025-12-10": 23933,
}

def add_bsr_data():
    """Add BSR data for both brands."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        # Create table with brand support
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manual_bsr (
                date TEXT,
                brand TEXT DEFAULT 'Trueseamoss',
                bsr REAL,
                PRIMARY KEY (date, brand)
            )
        """)
        
        # Check if brand column exists, if not, migrate
        cursor = conn.execute("PRAGMA table_info(manual_bsr)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "brand" not in columns:
            print("Migrating database to support brands...")
            # Backup existing data
            existing_data = conn.execute("SELECT date, bsr FROM manual_bsr").fetchall()
            
            # Create new table
            conn.execute("DROP TABLE manual_bsr")
            conn.execute("""
                CREATE TABLE manual_bsr (
                    date TEXT,
                    brand TEXT DEFAULT 'Trueseamoss',
                    bsr REAL,
                    PRIMARY KEY (date, brand)
                )
            """)
            
            # Migrate existing data to Trueseamoss
            for date, bsr in existing_data:
                conn.execute(
                    "INSERT INTO manual_bsr(date, brand, bsr) VALUES(?, ?, ?)",
                    (date, "Trueseamoss", bsr)
                )
            conn.commit()
            print("Migration complete!")
        
        # Add Trueseamoss BSR data
        print("\nAdding Trueseamoss BSR data...")
        for date, bsr in trueseamoss_bsr.items():
            conn.execute(
                "INSERT OR REPLACE INTO manual_bsr(date, brand, bsr) VALUES(?, ?, ?)",
                (date, "Trueseamoss", float(bsr))
            )
            print(f"  {date}: {bsr}")
        
        # Add HerbalVineyard BSR data
        print("\nAdding HerbalVineyard BSR data...")
        for date, bsr in herbalvineyard_bsr.items():
            conn.execute(
                "INSERT OR REPLACE INTO manual_bsr(date, brand, bsr) VALUES(?, ?, ?)",
                (date, "HerbalVineyard", float(bsr))
            )
            print(f"  {date}: {bsr}")
        
        conn.commit()
        print("\n✅ All BSR data added successfully!")
        print("\n⚠️  Note: You may need to refresh the Streamlit app or wait ~60 seconds")
        print("   for the cache to expire to see the new BSR data in the dashboard.")

if __name__ == "__main__":
    print("Adding BSR data...")
    add_bsr_data()
    print("\nDone!")