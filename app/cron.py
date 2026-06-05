import asyncio
import json
import io
from PIL import Image, ImageDraw
from huggingface_hub import HfApi
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import TelemetryComponent
from app.config import settings

# Global dataset cache initialized from main application
hf_source_cache = None 

CLASS_MAPPING = {"BACKGROUND": 0, "PIT": 1, "STONE": 2, "CRATER": 3}

def render_mask(vector_str: str) -> bytes:
    """Renders categorical mask directly into image bytes."""
    mask = Image.new('L', (256, 256), color=0)
    draw = ImageDraw.Draw(mask)
    
    try:
        features = json.loads(vector_str)
        for f in features:
            points = [(p[0], p[1]) for p in f.get("points", [])]
            if len(points) >= 3:
                val = CLASS_MAPPING.get(f.get("class"), 0)
                draw.polygon(points, fill=val, outline=val)
    except Exception:
        pass
        
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()

def initialize_hf_dataset_repo():
    """Checks if the dataset repo exists, and creates it with a README.md if missing."""
    if not settings.HF_TOKEN:
        print("[INIT_HF] Skipping dataset repo init: HF_TOKEN is empty.")
        return
        
    api = HfApi(token=settings.HF_TOKEN)
    target_repo = "F1nnSBK/lunar-debris-and-voids"
    
    try:
        api.repo_info(repo_id=target_repo, repo_type="dataset")
        print(f"[INIT_HF] Dataset repository '{target_repo}' already exists.")
    except Exception:
        print(f"[INIT_HF] Repository '{target_repo}' does not exist. Creating...")
        try:
            api.create_repo(repo_id=target_repo, repo_type="dataset", private=False)
            
            # Upload basic README.md to describe structure
            readme_content = """---
license: mit
task_categories:
- image-segmentation
tags:
- lunar
- geology
- rocks
- craters
pretty_name: Lunar Debris and Voids Dataset
configs:
- config_name: default
  data_files:
  - split: train
    path: data/metadata_*.jsonl
---

# Lunar Debris and Voids Dataset

This dataset contains NAC lunar imagery labeled with high-precision instance segmentation masks of pits, stones, and craters.
It is actively populated by the `luna_labeler` tool.

## Dataset Structure

- `data/images/`: Raw 256x256 image patches.
- `data/masks/`: Rendered categorical masks (PNG format).
- `data/metadata_*.jsonl`: JSON lines containing bbox coordinates, class labels, and vector points.
"""
            api.upload_file(
                path_or_fileobj=readme_content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=target_repo,
                repo_type="dataset"
            )
            print("[INIT_HF] Dataset repository created and initialized with README.md successfully.")
        except Exception as e:
            print(f"[INIT_HF_ERROR] Failed to initialize repository: {e}")

def run_sync_cycle(db: Session) -> int:
    """Executes a single cycle of syncing verified, unsynced items to Hugging Face."""
    if not settings.HF_TOKEN:
        print("[SYNC] Skipping sync cycle: HF_TOKEN is empty.")
        return 0
        
    api = HfApi(token=settings.HF_TOKEN)
    target_repo = "F1nnSBK/lunar-debris-and-voids"
    
    unsynced = db.query(TelemetryComponent).filter_by(
        validation_status="VERIFIED", 
        synced_to_hf=False
    ).limit(50).all()

    if not unsynced or not hf_source_cache:
        return 0

    metadata_lines = []
    
    for item in unsynced:
        split_name, row_idx = item.file_path.split("::")
        
        # 1. Fetch original image from memory cache
        orig_img = hf_source_cache[split_name][int(row_idx)]["image"]
        img_buf = io.BytesIO()
        orig_img.save(img_buf, format="PNG")
        
        # 2. Render training target mask
        mask_bytes = render_mask(item.spatial_vector_data)
        
        # 3. Stream files directly to Hugging Face
        img_path = f"data/images/{item.id}.png"
        mask_path = f"data/masks/{item.id}.png"
        
        api.upload_file(path_or_fileobj=img_buf.getvalue(), path_in_repo=img_path, repo_id=target_repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=mask_bytes, path_in_repo=mask_path, repo_id=target_repo, repo_type="dataset")
        
        # 4. Append metadata generation line
        metadata_lines.append({
            "file_name": img_path,
            "mask_file_name": mask_path,
            "dominant_class": item.matrix_class,
            "operator_session": item.session_id,
            "raw_vector_data": item.spatial_vector_data
        })
        
        item.synced_to_hf = True
    
    # Append metadata chunk to the dataset repository
    jsonl_content = "\n".join([json.dumps(l) for l in metadata_lines]) + "\n"
    api.upload_file(
        path_or_fileobj=jsonl_content.encode("utf-8"),
        path_in_repo=f"data/metadata_{unsynced[0].id}.jsonl",
        repo_id=target_repo,
        repo_type="dataset"
    )
    
    db.commit()
    print(f"[SYNC] Successfully uploaded {len(unsynced)} components to HF.")
    return len(unsynced)

async def sync_supabase_to_huggingface():
    """Infinite loop executing every 30 minutes to push delta changes to HF."""
    # Ensure repository is initialized before starting loop
    initialize_hf_dataset_repo()
    
    while True:
        db: Session = SessionLocal()
        try:
            run_sync_cycle(db)
        except Exception as e:
            db.rollback()
            print(f"[SYNC_ERROR] Automation failed: {e}")
        finally:
            db.close()
            
        await asyncio.sleep(1800) # Sync intervals: 30 minutes
