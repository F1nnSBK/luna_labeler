import asyncio
import json
import io
import re
import os
import hashlib
from typing import List, Tuple
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, CommitOperationDelete, CommitOperationAdd
from huggingface_hub.utils import RepositoryNotFoundError
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import TelemetryComponent
from app.config import settings
from collections import namedtuple
from datetime import datetime

# Global dataset cache reference initialized from main application
hf_source_cache = None 

SyncResult = namedtuple("SyncResult", ["synced_images", "synced_annotations", "errors"])

CLASS_MAPPING = {"PIT": 1, "STONE": 2, "CRATER": 3}

def assign_split(image_id_str: str, seed: str = "luna_v2") -> str:
    """Deterministically assigns train/val/test splits based on image ID."""
    h = int(hashlib.md5(f"{seed}_{image_id_str}".encode()).hexdigest(), 16)
    r = h % 100
    if r < 70:
        return "train"
    elif r < 85:
        return "val"
    else:
        return "test"

def polygon_to_bbox_area(coords: List[float]) -> Tuple[List[float], float]:
    """
    coords: flat [x1, y1, x2, y2, ...] in COCO format
    returns: (bbox [x, y, w, h], area float)
    """
    pts = np.array(coords).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
    # Shoelace formula
    x, y = pts[:, 0], pts[:, 1]
    area = float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return bbox, area

def validate_coco(coco: dict) -> None:
    """Performs validation assertions on the generated COCO structure."""
    image_ids = {img["id"] for img in coco["images"]}
    category_ids = {cat["id"] for cat in coco["categories"]}
    ann_ids = set()

    for ann in coco["annotations"]:
        assert ann["id"] not in ann_ids,          f"Duplicate annotation id {ann['id']}"
        assert ann["image_id"] in image_ids,      f"Orphan annotation {ann['id']}"
        assert ann["category_id"] in category_ids, f"Unknown category {ann['category_id']}"
        assert ann["area"] > 0,                   f"Zero-area annotation {ann['id']}"
        assert len(ann["segmentation"][0]) >= 6,  f"Polygon too short {ann['id']}"
        assert ann["iscrowd"] == 0
        ann_ids.add(ann["id"])

def parse_nac_id(filename: str) -> str:
    """Parses LROC NAC ID from the filename using regex."""
    if not filename:
        return "UNKNOWN"
    base = filename.split("/")[-1]
    name, _ = os.path.splitext(base)
    match = re.search(r'(M\d+[LR][CE])', name)
    if match:
        return match.group(1)
    # Fallback to taking the last component after underscore
    parts = name.split('_')
    if parts:
        return parts[-1]
    return "UNKNOWN"

def initialize_hf_dataset_repo():
    """Checks if the dataset repo exists, wipes old structures if present, and creates README.md."""
    if not settings.HF_TOKEN:
        print("[INIT_HF] Skipping dataset repo init: HF_TOKEN is empty.")
        return
        
    api = HfApi(token=settings.HF_TOKEN)
    target_repo = "F1nnSBK/lunar-debris-and-voids"
    
    repo_exists = False
    try:
        api.repo_info(repo_id=target_repo, repo_type="dataset")
        repo_exists = True
        print(f"[INIT_HF] Dataset repository '{target_repo}' already exists.")
        
        # Check if the repo has already been converted to COCO v2
        files = api.list_repo_files(repo_id=target_repo, repo_type="dataset")
        if "annotations/instances_train.json" in files:
            print("[INIT_HF] Repository is already in COCO v2 format. Skipping initialization/wiping.")
            return
            
        # Wipe old legacy files if present
        files_to_delete = [f for f in files if f.startswith("data/") or f == "metadata.jsonl" or f == "README.md"]
        if files_to_delete:
            print(f"[INIT_HF] Wiping old legacy dataset files: {files_to_delete}")
            operations = [CommitOperationDelete(path_in_repo=f) for f in files_to_delete]
            api.create_commit(
                repo_id=target_repo,
                operations=operations,
                commit_message="Wipe old dataset files for COCO conversion",
                repo_type="dataset"
            )
    except RepositoryNotFoundError:
        print(f"[INIT_HF] Repository '{target_repo}' does not exist. Creating...")
    except Exception as e:
        print(f"[INIT_HF] Error fetching repo info: {e}. Attempting repo creation fallback...")

    if not repo_exists:
        try:
            api.create_repo(repo_id=target_repo, repo_type="dataset", private=False)
            print(f"[INIT_HF] Created new repository: {target_repo}")
        except Exception as e:
            print(f"[INIT_HF_ERROR] Failed to create repository: {e}")
            return

            
    # Upload clean YAML README.md
    readme_content = """---
license: cc-by-4.0
task_categories:
  - image-segmentation
task_ids:
  - instance-segmentation
annotations_creators:
  - expert-generated
language_creators:
  - found
language: []
multilinguality: []
size_categories:
  - 1K<n<10K
source_datasets:
  - original
tags:
  - lunar
  - remote-sensing
  - LROC
  - geology
  - pit-detection
  - COCO
dataset_info:
  features:
    - name: image
      dtype: image
    - name: annotations
      dtype: string
      description: COCO-format annotation JSON
  splits:
    - name: train
      num_examples: TBD
    - name: val
      num_examples: TBD
    - name: test
      num_examples: TBD
---

# Lunar Debris and Voids Dataset (v2.0.0)

## Dataset Description
This dataset contains expert-annotated high-precision segmentations of geological features (pits, stones, and craters) on the lunar surface, derived from LROC NAC (Lunar Reconnaissance Orbiter Camera Narrow Angle Camera) orbital imagery.

## Source Imagery
Primary imagery sourced from the Lunar Reconnaissance Orbiter (LRO) Narrow Angle Camera (NAC). Each patch has unique coordinates and source frame information.

## Class Definitions & Encoding
- Category 1: `pit` (geological_feature)
- Category 2: `stone` (geological_feature)
- Category 3: `crater` (geological_feature)

## Annotation Methodology
Annotations were created using MobileSAM-assisted polygon segmentation coupled with manual correction and expert validation.

## Split Strategy
Splits are assigned deterministically per-image using MD5 hashing of the image ID (70% train, 15% validation, 15% test).

## Known Limitations
Includes class imbalances and resolution variations inherent in remote-sensing datasets.

## Citation Instructions
Please cite Finn Hertsch, DHBW Ravensburg (2025) when referencing this dataset.
"""
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=target_repo,
            repo_type="dataset"
        )
        print("[INIT_HF] Uploaded README.md successfully.")
    except Exception as e:
        print(f"[INIT_HF_ERROR] Failed to upload README.md: {e}")

async def sync_to_hf(db: Session, hf_token: str, repo_id: str) -> SyncResult:
    """
    COCO dataset sync:
    1. Query all VERIFIED annotations from DB.
    2. Setup provenance and split assignments, flushing to DB immediately.
    3. Generate and batch instances_{split}.json.
    4. Fetch and batch pending images.
    5. Perform a single commit operation to Hugging Face.
    """
    if not hf_token:
        return SyncResult(0, 0, ["HF_TOKEN is empty."])

    api = HfApi(token=hf_token)
    
    # 1. Query all VERIFIED components sorted by ID to keep mapping stable
    components = db.query(TelemetryComponent).filter(
        TelemetryComponent.validation_status == "VERIFIED"
    ).order_by(TelemetryComponent.id).all()
    
    if not components:
        return SyncResult(0, 0, [])

    # Setup database helper for split & metadata setup
    for comp in components:
        # Check split
        if comp.hf_split is None:
            comp.hf_split = assign_split(str(comp.id))
        
        # Resolve missing provenance attributes
        if not comp.nac_id:
            try:
                split_name, row_idx = comp.file_path.split("::")
                row_idx = int(row_idx)
                image_obj = hf_source_cache[split_name][row_idx]["image"]
                filename = getattr(image_obj, "filename", "")
                comp.nac_id = parse_nac_id(filename)
            except Exception:
                comp.nac_id = "UNKNOWN"
                
        if comp.patch_origin_x is None:
            comp.patch_origin_x = 0
        if comp.patch_origin_y is None:
            comp.patch_origin_y = 0
            
    # Persist the updates immediately before uploading files
    if asyncio.iscoroutinefunction(db.flush):
        await db.flush()
    else:
        db.flush()

    # Build COCO datasets per split
    splits = ["train", "val", "test"]
    coco_images = {s: [] for s in splits}
    coco_annotations = {s: [] for s in splits}
    
    image_counter = 1
    annotation_counter = 1
    
    synced_images = 0
    synced_annotations = 0
    errors = []
    
    operations = []
    images_to_mark_synced = []
    staged_image_paths = set()

    for comp in components:
        split = comp.hf_split or "train"
        nac_id = comp.nac_id or "UNKNOWN"
        origin_x = comp.patch_origin_x or 0
        origin_y = comp.patch_origin_y or 0
        gsd = comp.gsd_m_per_px
        
        image_filename = f"patch_{nac_id}_{origin_x:06d}_{origin_y:06d}.png"
        image_id = image_counter
        image_counter += 1
        
        # Add image entry
        coco_images[split].append({
            "id": image_id,
            "file_name": image_filename,
            "width": 256,
            "height": 256,
            "license": 1,
            "nac_id": nac_id,
            "patch_origin_x": origin_x,
            "patch_origin_y": origin_y,
            "gsd_m_per_px": gsd
        })
        
        # Parse annotations if present
        if comp.spatial_vector_data:
            try:
                polys = json.loads(comp.spatial_vector_data)
                for poly in polys:
                    poly_class = poly.get("class", "UNKNOWN").upper()
                    cat_id = CLASS_MAPPING.get(poly_class)
                    if not cat_id:
                        # Skip unknown classes or background
                        continue
                        
                    points = poly.get("points", [])
                    if len(points) < 3:
                        continue
                        
                    flat_coords = []
                    for pt in points:
                        flat_coords.append(float(pt[0]))
                        flat_coords.append(float(pt[1]))
                        
                    bbox, area = polygon_to_bbox_area(flat_coords)
                    if area <= 0:
                        continue
                        
                    coco_annotations[split].append({
                        "id": annotation_counter,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "segmentation": [flat_coords],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                        "annotator_session": comp.session_id or "api_worker",
                        "annotation_mode": comp.annotation_mode or "sam_assisted"
                    })
                    annotation_counter += 1
                    
            except Exception as e:
                errors.append(f"Failed parsing annotations for {comp.id}: {e}")

        # Queue image upload if not synced
        if comp.hf_sync_status != "synced":
            path_in_repo = f"images/{split}/{image_filename}"
            if path_in_repo not in staged_image_paths:
                staged_image_paths.add(path_in_repo)
                try:
                    split_name, row_idx = comp.file_path.split("::")
                    row_idx = int(row_idx)
                    image_obj = hf_source_cache[split_name][row_idx]["image"]
                    
                    img_buf = io.BytesIO()
                    image_obj.save(img_buf, format="PNG")
                    
                    operations.append(CommitOperationAdd(
                        path_in_repo=path_in_repo,
                        path_or_fileobj=img_buf.getvalue()
                    ))
                    images_to_mark_synced.append(comp)
                except Exception as e:
                    errors.append(f"Failed staging image {comp.id}: {e}")
            else:
                # Already staged by a previous component sharing this patch, mark synced on success
                images_to_mark_synced.append(comp)


    # Validate and stage COCO JSON for each split
    for split in splits:
        coco_dict = {
            "info": {
                "description": "Lunar pit, stone, and crater segmentation dataset derived from LROC NAC imagery",
                "url": f"https://huggingface.co/datasets/{repo_id}",
                "version": "2.0.0",
                "year": 2025,
                "contributor": "Finn Hertsch, DHBW Ravensburg",
                "date_created": datetime.utcnow().strftime("%Y-%m-%d")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Creative Commons Attribution 4.0 International",
                    "url": "https://creativecommons.org/licenses/by/4.0/"
                }
            ],
            "categories": [
                { "id": 1, "name": "pit",     "supercategory": "geological_feature" },
                { "id": 2, "name": "stone",   "supercategory": "geological_feature" },
                { "id": 3, "name": "crater",  "supercategory": "geological_feature" }
            ],
            "images": coco_images[split],
            "annotations": coco_annotations[split]
        }
        
        try:
            # Assert schema validity
            validate_coco(coco_dict)
            
            coco_json_bytes = json.dumps(coco_dict, indent=2).encode("utf-8")
            operations.append(CommitOperationAdd(
                path_in_repo=f"annotations/instances_{split}.json",
                path_or_fileobj=coco_json_bytes
            ))
            synced_annotations += len(coco_annotations[split])
        except AssertionError as ae:
            errors.append(f"COCO validation failed for {split}: {ae}")
            return SyncResult(0, 0, errors)

    # Perform a single batch commit for all changes
    if operations:
        try:
            print(f"[SYNC] Committing {len(operations)} file operations (JSONs + images) to Hugging Face...")
            api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message="Sync COCO instances and images to HF",
                repo_type="dataset"
            )
            # Mark synced only after commit succeeds
            for comp in images_to_mark_synced:
                comp.hf_sync_status = "synced"
                synced_images += 1
        except Exception as e:
            errors.append(f"Hugging Face batch commit failed: {e}")
            return SyncResult(0, 0, errors)

    # Commit state changes
    if asyncio.iscoroutinefunction(db.commit):
        await db.commit()
    else:
        db.commit()

    return SyncResult(synced_images, synced_annotations, errors)

async def sync_supabase_to_huggingface():
    """Infinite loop executing every 30 minutes to push delta changes to HF."""
    initialize_hf_dataset_repo()
    
    while True:
        db = SessionLocal()
        try:
            target_repo = "F1nnSBK/lunar-debris-and-voids"
            # Execute sync_to_hf
            res = await sync_to_hf(db, settings.HF_TOKEN, target_repo)
            if res.errors:
                print(f"[SYNC_ERROR] Automation errors: {res.errors}")
            else:
                print(f"[SYNC] Successfully synced {res.synced_images} images and {res.synced_annotations} annotations.")
        except Exception as e:
            db.rollback()
            print(f"[SYNC_ERROR] Automation failed: {e}")
        finally:
            db.close()
            
        await asyncio.sleep(1800) # Sync intervals: 30 minutes
