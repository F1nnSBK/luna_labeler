import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DB_URL = os.getenv("TELEMETRY_DB_URL")
engine = create_engine(DB_URL)

with engine.connect() as conn:
    print("--- Count of local files in DB ---")
    cnt = conn.execute(text("SELECT count(*) FROM telemetry_components WHERE file_path NOT LIKE '%::%'")).scalar()
    print("count:", cnt)
    
    print("--- Count of total files ---")
    total = conn.execute(text("SELECT count(*) FROM telemetry_components")).scalar()
    print("total:", total)
