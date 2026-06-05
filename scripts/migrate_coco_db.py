import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Ensure local app path is visible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

DB_URL = os.getenv("TELEMETRY_DB_URL")
if not DB_URL:
    print("ERROR: TELEMETRY_DB_URL environment variable is missing.")
    sys.exit(1)

engine = create_engine(DB_URL)

def migrate():
    columns_to_add = [
        ("nac_id", "VARCHAR(32)"),
        ("patch_origin_x", "INTEGER"),
        ("patch_origin_y", "INTEGER"),
        ("gsd_m_per_px", "FLOAT"),
        ("annotation_mode", "VARCHAR(16) DEFAULT 'sam_assisted'"),
        ("hf_sync_status", "VARCHAR(16) DEFAULT 'pending'"),
        ("hf_split", "VARCHAR(8)")
    ]
    
    with engine.begin() as conn:
        for col_name, col_type in columns_to_add:
            try:
                # PostgreSQL ALTER TABLE ADD COLUMN IF NOT EXISTS
                query = f"ALTER TABLE telemetry_components ADD COLUMN IF NOT EXISTS {col_name} {col_type};"
                conn.execute(text(query))
                print(f"Added column {col_name} or it already exists.")
            except Exception as e:
                print(f"Error adding column {col_name}: {e}")
                sys.exit(1)
                
    print("Migration executed successfully.")

if __name__ == "__main__":
    migrate()
