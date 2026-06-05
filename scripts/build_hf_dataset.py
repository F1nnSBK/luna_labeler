import os
import csv
import json
import uuid
from pathlib import Path
from huggingface_hub import HfApi
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models import TelemetryComponent

load_dotenv()

HF_TOKEN = os.getenv("TELEMETRY_TOKEN")
DB_URL = os.getenv("TELEMETRY_DB_URL")
DATASET_REPO = "F1nnSBK/lunar-telemetry-assets"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
hf_api = HfApi(token=HF_TOKEN)

def load_catalogs(csv_path: str, json_path: str) -> tuple[dict, dict]:
    pits = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pits[row['id']] = row

    with open(json_path, 'r', encoding='utf-8') as f:
        nacs = json.load(f)

    return pits, nacs

def upload_to_huggingface(local_path: Path, hf_path: str):
    hf_api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=hf_path,
        repo_id=DATASET_REPO,
        repo_type="dataset"
    )

def register_component(db, component_id: str, hf_path: str, is_anchor: bool, matrix_class: str):
    component = TelemetryComponent(
        id=component_id,
        file_path=hf_path,
        confidence_index=1.0 if is_anchor else 0.0,
        matrix_class=matrix_class,
        is_baseline_anchor=is_anchor,
        validation_status="VERIFIED" if is_anchor else "PENDING",
        session_id="SYSTEM_INIT"
    )
    db.add(component)

def process_dataset(image_dir: Path, csv_path: str, json_path: str):
    db = SessionLocal()
    pits, nac_mapping = load_catalogs(csv_path, json_path)
    
    try:
        # 1. Process known pits (Honeypots / Anchors)
        for pit_id, nac_list in nac_mapping.items():
            if pit_id not in pits:
                continue
                
            for nac_id in nac_list:
                local_img = image_dir / "anchors" / f"{nac_id}_{pit_id}.png"
                if not local_img.exists():
                    continue

                component_id = f"ANC_{pit_id}_{nac_id}"
                hf_path = f"images/anchors/{component_id}.png"
                
                upload_to_huggingface(local_img, hf_path)
                register_component(db, component_id, hf_path, is_anchor=True, matrix_class="PIT")
                print(f"Registered Anchor: {component_id}")

        # 2. Process unknown anomalies from Spark
        anomaly_dir = image_dir / "anomalies"
        if anomaly_dir.exists():
            for local_img in anomaly_dir.glob("*.png"):
                nac_id = local_img.stem.split('_')[0] 
                short_hash = str(uuid.uuid4())[:6]
                
                component_id = f"TEL_{nac_id}_{short_hash}"
                hf_path = f"images/anomalies/{component_id}.png"
                
                upload_to_huggingface(local_img, hf_path)
                register_component(db, component_id, hf_path, is_anchor=False, matrix_class="UNKNOWN")
                print(f"Registered Anomaly: {component_id}")

        db.commit()
        print("Dataset successfully synchronized with Hugging Face and Supabase.")

    except Exception as e:
        db.rollback()
        print(f"Sync failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    process_dataset(
        image_dir=Path("./data/patches"),
        csv_path="./catalogs/lpa.csv",
        json_path="./catalogs/pit_nacs.json"
    )