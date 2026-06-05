#!/usr/bin/env python3
"""
Active Learning Dataset Compiler (Luna Labeler version)
Compiles a unified, scientifically sound, and COCO-standard dataset under data/active_learning_ds/
with a clean, unified naming system and detailed metadata (NAC ID, Pit Name) for all images.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("build_al_dataset")

# Paths to the sibling luna project
LUNA_ROOT = Path("/Users/finnhertsch/projects/luna")

# Mond/COCO config
COCO_CATEGORIES = [
    {"id": 1, "name": "pit", "supercategory": "geomorphology"}
]

def load_environment() -> dict:
    """Load environment variables from both local directory and luna_labeler directory."""
    env_vars = {}
    load_dotenv(ROOT / ".env")
    env_vars["TELEMETRY_TOKEN"] = os.getenv("TELEMETRY_TOKEN") or os.getenv("HF_TOKEN")
    env_vars["TELEMETRY_DB_URL"] = os.getenv("TELEMETRY_DB_URL")
    return env_vars

def parse_provenance(path_str: str) -> tuple[str, str]:
    """
    Parses LROC NAC ID and Pit Name from Hugging Face dataset image filename.
    Example positive: '.../pits/Adams_B_1_M1149067652RC.png' -> ('M1149067652RC', 'Adams_B_1')
    Example negative: '.../negatives/neg_0_M106088433LC.png' -> ('M106088433LC', '')
    """
    if not path_str or path_str == "None":
        return "UNKNOWN", ""
        
    filename = Path(path_str).name
    name, _ = os.path.splitext(filename)
    
    # Check for LROC pattern: M + digits + L/R + C/E
    # e.g., M1149067652RC
    match = re.search(r'(M\d+[LR][CE])', name)
    if match:
        nac_id = match.group(1)
        # Parse pit name if present before the NAC ID (minus any trailing underscore)
        prefix = name.split(nac_id)[0].rstrip("_")
        
        # If prefix is just "neg_0" or "neg", it's a negative, so no pit name
        if prefix.startswith("neg_") or prefix == "neg":
            return nac_id, ""
            
        return nac_id, prefix
        
    return "UNKNOWN", ""

def parse_local_filename(filename: str) -> tuple[str, str]:
    """
    Parses NAC ID from local pipeline file names.
    e.g. 'neg_active_M193046922_M193046922LC_rank001.png' -> ('M193046922LC', 'rank001')
    """
    name, _ = os.path.splitext(filename)
    match = re.search(r'(M\d+[LR][CE])', name)
    if match:
        nac_id = match.group(1)
        # Suffix is everything after the nac_id
        parts = name.split(nac_id)
        suffix = parts[-1].lstrip("_") if len(parts) > 1 else ""
        return nac_id, suffix
    return "UNKNOWN", ""

def fetch_positives_from_hf(token: str | None) -> list[dict]:
    """Downloads F1nnSBK/lunar-pits-dataset and filters for positives (label=1)."""
    dataset_id = "F1nnSBK/lunar-pits-dataset"
    log.info("Loading Hugging Face dataset %s ...", dataset_id)
    try:
        ds = load_dataset(dataset_id, token=token)
    except Exception as e:
        log.error("Failed to load dataset from Hugging Face: %s", e)
        log.error("Please make sure TELEMETRY_TOKEN is correct and has access to the repository.")
        sys.exit(1)
        
    positives = []
    for split in ds.keys():
        log.info("Filtering split '%s' for positives...", split)
        for idx, item in enumerate(ds[split]):
            # Label 1 is pits
            if item.get("label") == 1:
                img = item["image"]
                hf_path = getattr(img, "filename", "None")
                nac_id, pit_name = parse_provenance(hf_path)
                
                # Unified naming convention: pos_{nac_id}_{pit_name}.png
                # If no pit_name is parsed, fallback to index
                pit_suffix = f"_{pit_name}" if pit_name else f"_hf_{split}_{idx:04d}"
                file_name = f"pos_{nac_id}{pit_suffix}.png"
                
                positives.append({
                    "image": img,
                    "source": f"hf_{split}_{idx}",
                    "file_name": file_name,
                    "nac_id": nac_id,
                    "pit_name": pit_name,
                    "width": 256,
                    "height": 256,
                    "status": "positive"
                })
    log.info("Successfully fetched %d positives from Hugging Face.", len(positives))
    return positives

def gather_local_negatives(negatives_dirs: list[Path], temp_dir: Path) -> list[dict]:
    """Scans vit_dataset/negatives and temp directory in luna repository for negative PNGs."""
    negatives = []
    seen_files = set()
    
    # Check negatives directories
    for n_dir in negatives_dirs:
        if n_dir.exists():
            log.info("Scanning negatives directory %s ...", n_dir)
            for file_path in n_dir.glob("*.png"):
                if file_path.name in seen_files:
                    continue
                seen_files.add(file_path.name)
                
                # Parse NAC ID from filename
                nac_id, suffix = parse_local_filename(file_path.name)
                suffix_str = f"_{suffix}" if suffix else f"_{file_path.stem}"
                file_name = f"neg_{nac_id}{suffix_str}.png"
                
                try:
                    with Image.open(file_path) as img:
                        w, h = img.size
                    negatives.append({
                        "file_path": file_path,
                        "source": "local_negatives",
                        "file_name": file_name,
                        "nac_id": nac_id,
                        "pit_name": "",
                        "width": w,
                        "height": h,
                        "status": "negative"
                    })
                except Exception as e:
                    log.warning("Could not read image %s: %s", file_path, e)
                    
    # Scan temp directory for negative patterns
    if temp_dir.exists():
        log.info("Scanning temp directory %s ...", temp_dir)
        for file_path in temp_dir.glob("*.png"):
            if file_path.name in seen_files:
                continue
            if file_path.name.startswith("neg_") or "negative" in file_path.name:
                seen_files.add(file_path.name)
                
                nac_id, suffix = parse_local_filename(file_path.name)
                suffix_str = f"_{suffix}" if suffix else f"_{file_path.stem}"
                file_name = f"neg_{nac_id}{suffix_str}.png"
                
                try:
                    with Image.open(file_path) as img:
                        w, h = img.size
                    negatives.append({
                        "file_path": file_path,
                        "source": "temp_negatives",
                        "file_name": file_name,
                        "nac_id": nac_id,
                        "pit_name": "",
                        "width": w,
                        "height": h,
                        "status": "negative"
                    })
                except Exception as e:
                    log.warning("Could not read image %s: %s", file_path, e)
                    
    log.info("Successfully gathered %d local negative tiles.", len(negatives))
    return negatives

def fetch_from_db(db_url: str | None, token: str | None) -> tuple[list[dict], list[dict], list[dict]]:
    """Queries telemetry_components database for verified positives, verified negatives, and pending potentials."""
    if not db_url:
        log.warning("No database URL available. Skipping database sync.")
        return [], [], []
        
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
    except ImportError:
        log.warning("SQLAlchemy not installed. Skipping database sync.")
        return [], [], []
        
    log.info("Querying Supabase database for telemetry components...")
    
    positives = []
    negatives = []
    potentials = []
    
    try:
        # Load dataset cache for resolving file_path reference (e.g. train::12)
        log.info("Loading HF dataset cache for DB reference resolution...")
        hf_cache = load_dataset("F1nnSBK/lunar-pits-dataset", token=token)
        
        with engine.connect() as conn:
            query = "SELECT id, file_path, matrix_class, validation_status, nac_id FROM telemetry_components"
            rows = conn.execute(text(query)).fetchall()
            
            for row in rows:
                comp_id = row[0]
                file_path = row[1]
                matrix_class = row[2] or "UNKNOWN"
                validation_status = row[3] or "PENDING"
                db_nac_id = row[4]
                
                # Check if file_path is of format split::idx (Hugging Face cache reference)
                if "::" in file_path:
                    try:
                        split, idx_str = file_path.split("::")
                        idx = int(idx_str)
                        hf_item = hf_cache[split][idx]
                        img = hf_item["image"]
                        hf_path = getattr(img, "filename", "None")
                        
                        # Parse coordinates/provenance
                        nac_id, pit_name = parse_provenance(hf_path)
                        nac_id = db_nac_id or nac_id
                        
                        # Set standardized filename
                        if validation_status == "VERIFIED":
                            if matrix_class in ["PIT", "STONE", "CRATER"]:
                                pit_suffix = f"_{pit_name}" if pit_name else f"_{matrix_class.lower()}_db_{comp_id}"
                                file_name = f"pos_{nac_id}{pit_suffix}.png"
                                
                                positives.append({
                                    "image": img,
                                    "source": f"db_{comp_id}",
                                    "file_name": file_name,
                                    "nac_id": nac_id,
                                    "pit_name": pit_name,
                                    "width": 256,
                                    "height": 256,
                                    "status": "positive"
                                })
                            else:
                                file_name = f"neg_{nac_id}_db_{comp_id}.png"
                                negatives.append({
                                    "image": img,
                                    "source": f"db_{comp_id}",
                                    "file_name": file_name,
                                    "nac_id": nac_id,
                                    "pit_name": "",
                                    "width": 256,
                                    "height": 256,
                                    "status": "negative"
                                })
                        elif validation_status == "PENDING":
                            file_name = f"potential_{nac_id}_db_{comp_id}.png"
                            potentials.append({
                                "image": img,
                                "source": f"db_{comp_id}",
                                "file_name": file_name,
                                "nac_id": nac_id,
                                "pit_name": "",
                                "width": 256,
                                "height": 256,
                                "status": "potential"
                            })
                            
                    except Exception as err:
                        log.debug("Error resolving database file_path %s: %s", file_path, err)
                else:
                    # This is a local file path entry
                    try:
                        # Extract nac_id from filename or DB column
                        path_obj = Path(file_path)
                        nac_id, suffix = parse_local_filename(path_obj.name)
                        nac_id = db_nac_id or nac_id
                        
                        # Load image to read dims
                        abs_path = path_obj if path_obj.is_absolute() else (ROOT / path_obj)
                        if not abs_path.exists():
                            abs_path = LUNA_ROOT / file_path
                        if not abs_path.exists():
                            continue
                            
                        with Image.open(abs_path) as img:
                            w, h = img.size
                            
                        if validation_status == "VERIFIED":
                            if matrix_class in ["PIT", "STONE", "CRATER"]:
                                file_name = f"pos_{nac_id}_db_{comp_id}.png"
                                positives.append({
                                    "file_path": abs_path,
                                    "source": f"db_{comp_id}",
                                    "file_name": file_name,
                                    "nac_id": nac_id,
                                    "pit_name": "",
                                    "width": w,
                                    "height": h,
                                    "status": "positive"
                                })
                            else:
                                file_name = f"neg_{nac_id}_db_{comp_id}.png"
                                negatives.append({
                                    "file_path": abs_path,
                                    "source": f"db_{comp_id}",
                                    "file_name": file_name,
                                    "nac_id": nac_id,
                                    "pit_name": "",
                                    "width": w,
                                    "height": h,
                                    "status": "negative"
                                })
                        elif validation_status == "PENDING":
                            file_name = f"potential_{nac_id}_db_{comp_id}.png"
                            potentials.append({
                                "file_path": abs_path,
                                "source": f"db_{comp_id}",
                                "file_name": file_name,
                                "nac_id": nac_id,
                                "pit_name": "",
                                "width": w,
                                "height": h,
                                "status": "potential"
                            })
                    except Exception as err:
                        log.debug("Error loading local DB component %s: %s", file_path, err)
                        
        log.info("Supabase Sync: Found %d positives, %d negatives, %d potentials.", 
                 len(positives), len(negatives), len(potentials))
    except Exception as e:
        log.error("Database query failed: %s", e)
        
    return positives, negatives, potentials

def compile_coco(images_list: list[dict], with_annotations: bool = True) -> dict:
    """Builds a COCO JSON structure from list of image dicts with unified metadata."""
    coco = {
        "info": {
            "description": "Luna Active Learning Dataset",
            "version": "1.1.0",
            "year": 2026,
            "contributor": "Finn Hertsch",
            "date_created": "2026-06-05"
        },
        "licenses": [],
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": []
    }
    
    next_img_id = 1
    next_ann_id = 1
    
    for item in images_list:
        img_id = next_img_id
        next_img_id += 1
        
        # Add image entry with unified metadata
        coco["images"].append({
            "id": img_id,
            "file_name": item["file_name"],
            "width": item["width"],
            "height": item["height"],
            "nac_id": item["nac_id"],
            "pit_name": item["pit_name"],
            "status": item["status"],
            "source": item["source"]
        })
        
        # Add annotation if positive and annotations requested
        if with_annotations and item["status"] == "positive":
            bbox = [0, 0, item["width"], item["height"]]
            area = float(item["width"] * item["height"])
            
            coco["annotations"].append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": []  # empty polygon since we don't have segmentation masks
            })
            next_ann_id += 1
            
    return coco

def main() -> int:
    parser = argparse.ArgumentParser(description="Compile unified Active Learning COCO dataset.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "active_learning_ds",
                        help="Output directory for the compiled dataset")
    parser.add_argument("--query-db", action="store_true", default=False,
                        help="Query the Supabase labeler database to sync labels")
    args = parser.parse_args()
    
    # 1. Setup paths
    out_dir = args.out_dir
    img_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    
    # Clear directory if it exists to clean up legacy naming files
    if out_dir.exists():
        log.info("Clearing legacy compiled dataset directory %s...", out_dir)
        shutil.rmtree(out_dir)
        
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Target directory: %s", out_dir)
    
    # Load settings/credentials
    env = load_environment()
    token = env.get("TELEMETRY_TOKEN")
    db_url = env.get("TELEMETRY_DB_URL") if args.query_db else None
    
    # 2. Fetch/gather components
    # A. Positives from Hugging Face
    hf_positives = fetch_positives_from_hf(token)
    
    # B. Negatives from vit_dataset/negatives and temp/ in luna repository
    negatives_dirs = [
        LUNA_ROOT / "data" / "vit_dataset" / "negatives",
        LUNA_ROOT / "data" / "vit_dataset" / "negatives_active"
    ]
    temp_dir = LUNA_ROOT / "temp"
    local_negatives = gather_local_negatives(negatives_dirs, temp_dir)
    
    # C. Elements from database (optional)
    db_positives, db_negatives, db_potentials = fetch_from_db(db_url, token)
    
    # 3. Combine pools and filter duplicates by file_name to prevent double-saving
    all_positives = {}
    all_negatives = {}
    all_potentials = {}
    
    # Add HF positives
    for item in hf_positives:
        all_positives[item["file_name"]] = item
        
    # Add DB positives (overwriting or complementing)
    for item in db_positives:
        all_positives[item["file_name"]] = item
        
    # Add Local negatives
    for item in local_negatives:
        all_negatives[item["file_name"]] = item
        
    # Add DB negatives
    for item in db_negatives:
        all_negatives[item["file_name"]] = item
        
    # Add DB potentials
    for item in db_potentials:
        all_potentials[item["file_name"]] = item
        
    # Also look if there are local potentials
    local_potentials_dir = LUNA_ROOT / "data" / "vit_dataset" / "potentials"
    if local_potentials_dir.exists():
        log.info("Scanning local potentials directory %s ...", local_potentials_dir)
        for p_file in local_potentials_dir.glob("*.png"):
            nac_id, suffix = parse_local_filename(p_file.name)
            suffix_str = f"_{suffix}" if suffix else f"_{p_file.stem}"
            name = f"potential_{nac_id}{suffix_str}.png"
            if name not in all_potentials:
                try:
                    with Image.open(p_file) as img:
                        w, h = img.size
                    all_potentials[name] = {
                        "file_path": p_file,
                        "source": "local_potentials",
                        "file_name": name,
                        "nac_id": nac_id,
                        "pit_name": "",
                        "width": w,
                        "height": h,
                        "status": "potential"
                    }
                except Exception as e:
                    log.warning("Could not read image %s: %s", p_file, e)
                    
    positives_list = list(all_positives.values())
    negatives_list = list(all_negatives.values())
    potentials_list = list(all_potentials.values())
    
    log.info("Total compile count: positives=%d, negatives=%d, potentials=%d",
             len(positives_list), len(negatives_list), len(potentials_list))
             
    # 4. Save image files and build list of items
    final_items = []
    
    # Save positive images
    log.info("Saving positive images...")
    for idx, item in enumerate(positives_list):
        dest_path = img_dir / item["file_name"]
        if "image" in item:
            item["image"].save(dest_path)
        elif "file_path" in item:
            shutil.copy2(item["file_path"], dest_path)
        final_items.append(item)
        
    # Save negative images
    log.info("Saving negative images...")
    for idx, item in enumerate(negatives_list):
        dest_path = img_dir / item["file_name"]
        if "image" in item:
            item["image"].save(dest_path)
        elif "file_path" in item:
            shutil.copy2(item["file_path"], dest_path)
        final_items.append(item)
        
    # Save potential images
    log.info("Saving potential images...")
    for idx, item in enumerate(potentials_list):
        dest_path = img_dir / item["file_name"]
        if "image" in item:
            item["image"].save(dest_path)
        elif "file_path" in item:
            shutil.copy2(item["file_path"], dest_path)
        final_items.append(item)
        
    # 5. Build and write COCO JSON files
    log.info("Compiling COCO JSON files...")
    
    # A. labeled.json (Positives + Negatives)
    labeled_pool = [item for item in final_items if item["status"] in ["positive", "negative"]]
    labeled_coco = compile_coco(labeled_pool, with_annotations=True)
    (ann_dir / "labeled.json").write_text(json.dumps(labeled_coco, indent=2))
    
    # B. unlabeled.json (Potentials only)
    unlabeled_pool = [item for item in final_items if item["status"] == "potential"]
    unlabeled_coco = compile_coco(unlabeled_pool, with_annotations=False)
    (ann_dir / "unlabeled.json").write_text(json.dumps(unlabeled_coco, indent=2))
    
    # C. active_learning_pool.json (Master pool with metadata status)
    master_coco = compile_coco(final_items, with_annotations=True)
    (ann_dir / "active_learning_pool.json").write_text(json.dumps(master_coco, indent=2))
    
    log.info("COCO dataset generation complete! Output files:")
    log.info("  - Image directory: %s", img_dir)
    log.info("  - Labeled dataset annotations: %s", ann_dir / "labeled.json")
    log.info("  - Unlabeled dataset annotations: %s", ann_dir / "unlabeled.json")
    log.info("  - Master dataset annotations: %s", ann_dir / "active_learning_pool.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
