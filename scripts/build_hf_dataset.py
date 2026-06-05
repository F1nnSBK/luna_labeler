#!/usr/bin/env python3
"""
build_hf_dataset.py
Reads VERIFIED rows with real polygons from Supabase, builds a COCO-compliant
dataset, and uploads everything to HF in a single atomic commit.

Usage:
    python3 scripts/build_hf_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("build_hf_dataset")

# ── Config ────────────────────────────────────────────────────────────────────

DB_URL     = os.getenv("TELEMETRY_DB_URL")
HF_TOKEN   = os.getenv("TELEMETRY_TOKEN") or os.getenv("HF_TOKEN")
HF_REPO    = "F1nnSBK/lunar-debris-and-voids"
IMAGES_DIR = ROOT / "data" / "active_learning_ds" / "images"

if not DB_URL:
    log.error("TELEMETRY_DB_URL not set.")
    sys.exit(1)
if not HF_TOKEN:
    log.error("TELEMETRY_TOKEN / HF_TOKEN not set.")
    sys.exit(1)

COCO_INFO = {
    "description": "Lunar pit, stone and crater segmentation dataset — LROC NAC imagery",
    "url": f"https://huggingface.co/datasets/{HF_REPO}",
    "version": "2.0.0",
    "year": datetime.utcnow().year,
    "contributor": "Finn Hertsch, DHBW Ravensburg",
    "date_created": datetime.utcnow().strftime("%Y-%m-%d"),
}

COCO_LICENSES = [
    {"id": 1, "name": "Creative Commons Attribution 4.0 International",
     "url": "https://creativecommons.org/licenses/by/4.0/"}
]

COCO_CATEGORIES = [
    {"id": 1, "name": "pit",    "supercategory": "geological_feature"},
    {"id": 2, "name": "stone",  "supercategory": "geological_feature"},
    {"id": 3, "name": "crater", "supercategory": "geological_feature"},
]

CLASS_TO_CAT_ID = {"PIT": 1, "STONE": 2, "CRATER": 3}


# ── Helpers ───────────────────────────────────────────────────────────────────

def assign_split(row_id: str, seed: str = "luna_v2") -> str:
    h = int(hashlib.md5(f"{seed}_{row_id}".encode()).hexdigest(), 16)
    r = h % 100
    if r < 70:
        return "train"
    elif r < 85:
        return "val"
    return "test"


def polygon_to_bbox_area(coords: list[float]) -> tuple[list[float], float]:
    pts = np.array(coords).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox = [float(x_min), float(y_min),
            float(x_max - x_min), float(y_max - y_min)]
    x, y = pts[:, 0], pts[:, 1]
    area = float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return bbox, area


def parse_polygons(raw: list, matrix_class: str) -> list[dict]:
    """
    Handles all three formats spatial_vector_data can arrive in:

      1. Canvas format (luna_labeler UI output):
         [{"class": "PIT", "points": [[x,y], [x,y], ...]}, ...]

      2. List of flat coord arrays:
         [[x1,y1,x2,y2,...], [x1,y1,...]]

      3. Single flat coord array:
         [x1, y1, x2, y2, ...]

    Returns list of dicts: {"coords": [flat floats], "cat_id": int}
    """
    result = []

    if not raw:
        return result

    # Format 1: list of dicts with "points" key
    if isinstance(raw[0], dict):
        for obj in raw:
            pts = obj.get("points", [])
            cls = obj.get("class", matrix_class or "PIT").upper()
            flat = [coord for pt in pts for coord in (float(pt[0]), float(pt[1]))]
            if len(flat) >= 6:
                result.append({
                    "coords": flat,
                    "cat_id": CLASS_TO_CAT_ID.get(cls, 1),
                })
        return result

    # Format 2: list of lists
    if isinstance(raw[0], list):
        cat_id = CLASS_TO_CAT_ID.get((matrix_class or "PIT").upper(), 1)
        for poly in raw:
            flat = [float(v) for v in poly]
            if len(flat) >= 6:
                result.append({"coords": flat, "cat_id": cat_id})
        return result

    # Format 3: single flat array of numbers
    if isinstance(raw[0], (int, float)):
        cat_id = CLASS_TO_CAT_ID.get((matrix_class or "PIT").upper(), 1)
        flat = [float(v) for v in raw]
        if len(flat) >= 6:
            result.append({"coords": flat, "cat_id": cat_id})

    return result


def validate_coco(coco: dict) -> None:
    image_ids    = {img["id"] for img in coco["images"]}
    category_ids = {cat["id"] for cat in coco["categories"]}
    ann_ids: set[int] = set()
    for ann in coco["annotations"]:
        assert ann["id"] not in ann_ids,           f"Duplicate annotation id {ann['id']}"
        assert ann["image_id"] in image_ids,       f"Orphan annotation {ann['id']}"
        assert ann["category_id"] in category_ids, f"Unknown category {ann['category_id']}"
        assert ann["area"] > 0,                    f"Zero-area annotation {ann['id']}"
        assert len(ann["segmentation"][0]) >= 6,   f"Polygon too short {ann['id']}"
        assert ann["iscrowd"] == 0
        ann_ids.add(ann["id"])
    log.info("COCO validation passed: %d images, %d annotations",
             len(coco["images"]), len(coco["annotations"]))


def empty_coco(split: str) -> dict:
    return {
        "info": {**COCO_INFO, "description": f"{COCO_INFO['description']} — {split}"},
        "licenses": COCO_LICENSES,
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }


# ── DB ────────────────────────────────────────────────────────────────────────

def fetch_verified_rows(engine) -> list[dict]:
    sql = text("""
        SELECT id, file_path, matrix_class, spatial_vector_data,
               nac_id, patch_origin_x, patch_origin_y, gsd_m_per_px,
               annotation_mode, hf_split
        FROM telemetry_components
        WHERE validation_status = 'VERIFIED'
          AND spatial_vector_data IS NOT NULL
        ORDER BY id
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    log.info("Fetched %d VERIFIED rows with polygons from Supabase.", len(rows))
    return [dict(r) for r in rows]


def persist_splits(engine, id_split_pairs: list[tuple[str, str]]) -> None:
    sql = text("""
        UPDATE telemetry_components
        SET hf_split = :split, hf_sync_status = 'synced'
        WHERE id = :id
    """)
    with engine.begin() as conn:
        conn.execute(sql, [{"id": i, "split": s} for i, s in id_split_pairs])
    log.info("Persisted %d split assignments in Supabase.", len(id_split_pairs))


# ── Build ─────────────────────────────────────────────────────────────────────

def build_coco_splits(rows: list[dict]) -> tuple[dict, dict, dict, list[tuple[str, str]]]:
    splits = {s: empty_coco(s) for s in ("train", "val", "test")}
    image_counters = {"train": 0, "val": 0, "test": 0}
    ann_counter = 0
    id_split_pairs: list[tuple[str, str]] = []

    for row in rows:
        split = row["hf_split"] or assign_split(row["id"])
        id_split_pairs.append((row["id"], split))

        image_counters[split] += 1
        img_id = image_counters[split]

        splits[split]["images"].append({
            "id":             img_id,
            "file_name":      row["file_path"],
            "width":          256,
            "height":         256,
            "license":        1,
            "nac_id":         row.get("nac_id") or "UNKNOWN",
            "patch_origin_x": row.get("patch_origin_x"),
            "patch_origin_y": row.get("patch_origin_y"),
            "gsd_m_per_px":   row.get("gsd_m_per_px"),
        })

        # Parse spatial_vector_data — handles all three formats
        try:
            raw = json.loads(row["spatial_vector_data"])
        except (json.JSONDecodeError, TypeError):
            log.warning("Could not parse spatial_vector_data for %s — skipping.", row["file_path"])
            continue

        polys = parse_polygons(raw, row.get("matrix_class", "PIT"))

        for poly in polys:
            bbox, area = polygon_to_bbox_area(poly["coords"])
            if area <= 0:
                continue
            ann_counter += 1
            splits[split]["annotations"].append({
                "id":              ann_counter,
                "image_id":        img_id,
                "category_id":     poly["cat_id"],
                "segmentation":    [poly["coords"]],
                "area":            area,
                "bbox":            bbox,
                "iscrowd":         0,
                "annotation_mode": row.get("annotation_mode", "sam_assisted"),
            })

    return splits["train"], splits["val"], splits["test"], id_split_pairs


# ── Upload ────────────────────────────────────────────────────────────────────

def build_readme() -> str:
    return """\
---
license: cc-by-4.0
task_categories:
  - image-segmentation
task_ids:
  - instance-segmentation
annotations_creators:
  - expert-generated
tags:
  - lunar
  - remote-sensing
  - LROC
  - geology
  - pit-detection
  - COCO
---

# Lunar Debris and Voids

Instance segmentation dataset of geological features on the lunar surface,
derived from Lunar Reconnaissance Orbiter Camera (LROC) NAC imagery.

## Classes

| id | name   | description                            |
|----|--------|----------------------------------------|
| 1  | pit    | Volcanic collapse / lava tube skylight |
| 2  | stone  | Surface boulder / rock                 |
| 3  | crater | Impact crater rim / bowl               |

## Format

COCO Detection 2017. Annotations under `annotations/instances_{split}.json`.
Images under `images/{split}/`.

## Annotation methodology

Interactive segmentation via MobileSAM (box-prompt → polygon) with
manual fallback mode. Annotated using luna_labeler on LROC NAC patches
(256×256 px crops).

## Citation

Hertsch, F. (2026). *Luna: Lunar Pit Detection Pipeline*. DHBW Ravensburg.
"""


def upload_to_hf(train: dict, val: dict, test: dict, rows: list[dict]) -> None:
    api = HfApi(token=HF_TOKEN)
    ops: list[CommitOperationAdd] = []
    staged_images: set[str] = set()

    # Stage COCO JSONs
    for split_name, coco in [("train", train), ("val", val), ("test", test)]:
        validate_coco(coco)
        blob = json.dumps(coco, indent=2).encode("utf-8")
        ops.append(CommitOperationAdd(
            path_in_repo=f"annotations/instances_{split_name}.json",
            path_or_fileobj=BytesIO(blob),
        ))
        log.info("Staged annotations/instances_%s.json (%d images, %d annotations)",
                 split_name, len(coco["images"]), len(coco["annotations"]))

    # Stage README
    ops.append(CommitOperationAdd(
        path_in_repo="README.md",
        path_or_fileobj=BytesIO(build_readme().encode("utf-8")),
    ))

    # Build file_name → split lookup
    file_to_split: dict[str, str] = {}
    for split_name, coco in [("train", train), ("val", val), ("test", test)]:
        for img in coco["images"]:
            file_to_split[img["file_name"]] = split_name

    # Stage images
    for row in rows:
        file_name = row["file_path"]
        split = file_to_split.get(file_name)
        if not split:
            continue
        hf_path = f"images/{split}/{file_name}"
        if hf_path in staged_images:
            continue
        local_path = IMAGES_DIR / file_name
        if not local_path.exists():
            log.warning("Image not on disk, skipping: %s", file_name)
            continue
        ops.append(CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=local_path))
        staged_images.add(hf_path)

    log.info("Committing %d ops to HF (%d images)...", len(ops), len(staged_images))

    api.create_commit(
        repo_id=HF_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=(
            f"feat: COCO v2 — {len(staged_images)} images, "
            f"{sum(len(c['annotations']) for c in [train, val, test])} annotations"
        ),
    )
    log.info("HF commit successful.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    engine = create_engine(DB_URL, pool_pre_ping=True)

    rows = fetch_verified_rows(engine)
    if not rows:
        log.warning("No VERIFIED rows with polygons found. Annotate first, then re-run.")
        sys.exit(0)

    train, val, test, id_split_pairs = build_coco_splits(rows)
    log.info("Split summary: train=%d  val=%d  test=%d",
             len(train["images"]), len(val["images"]), len(test["images"]))

    upload_to_hf(train, val, test, rows)
    persist_splits(engine, id_split_pairs)
    log.info("All done.")


if __name__ == "__main__":
    main()