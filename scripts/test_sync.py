import os
import sys
import asyncio
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Ensure local app path is visible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.cron import sync_to_hf, initialize_hf_dataset_repo, hf_source_cache
from app.config import settings
from datasets import load_dataset

load_dotenv()

async def main():
    print("Initializing Hugging Face source cache in cron...")
    # Load dataset to populate cache
    ds_cache = load_dataset("F1nnSBK/lunar-pits-dataset", token=settings.HF_TOKEN)
    import app.cron as cron
    cron.hf_source_cache = ds_cache
    
    print("Calling initialize_hf_dataset_repo to ensure repo exists and is clean...")
    initialize_hf_dataset_repo()
    
    print("Starting sync cycle...")
    db = SessionLocal()
    try:
        res = await sync_to_hf(db, settings.HF_TOKEN, "F1nnSBK/lunar-debris-and-voids")
        print("Sync results:")
        print(f"Synced Images: {res.synced_images}")
        print(f"Synced Annotations: {res.synced_annotations}")
        print(f"Errors: {res.errors}")
    except Exception as e:
        print(f"Sync failed: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
