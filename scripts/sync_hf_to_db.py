import os
import uuid
from datasets import load_dataset
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models import TelemetryComponent

load_dotenv()

DB_URL = os.getenv("TELEMETRY_DB_URL")
HF_TOKEN = os.getenv("TELEMETRY_TOKEN")
DATASET_ID = "F1nnSBK/lunar-pits-dataset"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

def sync_existing_dataset():
    db = SessionLocal()
    
    try:
        dataset = load_dataset(DATASET_ID, token=HF_TOKEN)
        
        for split in dataset.keys():
            for idx, item in enumerate(dataset[split]):
                label_name = "PIT" if item["label"] == 1 else "UNKNOWN"
                is_anchor = label_name == "PIT"
                
                # Format: split_name::row_index (e.g., train::42)
                internal_path = f"{split}::{idx}"
                component_id = f"ANC_{str(uuid.uuid4())[:8]}"
                
                existing = db.query(TelemetryComponent).filter_by(file_path=internal_path).first()
                if existing:
                    continue

                component = TelemetryComponent(
                    id=component_id,
                    file_path=internal_path,
                    confidence_index=1.0 if is_anchor else 0.5,
                    matrix_class=label_name,
                    is_baseline_anchor=is_anchor,
                    validation_status="VERIFIED" if is_anchor else "PENDING",
                    session_id="SYSTEM_INIT"
                )
                db.add(component)
                
        db.commit()

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    sync_existing_dataset()