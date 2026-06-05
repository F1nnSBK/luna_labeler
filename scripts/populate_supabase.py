#!/usr/bin/env python3
"""
populate_supabase.py
Reads data/active_learning_ds/annotations/{labeled,unlabeled}.json and populates
the telemetry_components table in Supabase using a single bulk INSERT.

Run once after a full wipe:
    python3 scripts/populate_supabase.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("populate_supabase")

DB_URL = os.getenv("TELEMETRY_DB_URL")
if not DB_URL:
    log.error("TELEMETRY_DB_URL is not set. Aborting.")
    sys.exit(1)

ANNOTATIONS_DIR = ROOT / "data" / "active_learning_ds" / "annotations"
IMAGES_DIR      = ROOT / "data" / "active_learning_ds" / "images"


def load_coco(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_row(img: dict) -> dict | None:
    file_name  = img["file_name"]

    if not (IMAGES_DIR / file_name).exists():
        return None

    status_key   = img.get("status", "negative")
    is_positive  = status_key == "positive"
    short_id     = hashlib.md5(file_name.encode()).hexdigest()[:8]

    return {
        "id":                  short_id,
        "file_path":           file_name,
        "confidence_index":    0.5 if is_positive else 1.0,
        "matrix_class":        "PIT" if is_positive else "NEGATIVE",
        "is_baseline_anchor":  is_positive,
        "validation_status":   "PENDING",
        "nac_id":              img.get("nac_id", "UNKNOWN"),
        "patch_origin_x":      img.get("patch_origin_x"),
        "patch_origin_y":      img.get("patch_origin_y"),
        "gsd_m_per_px":        img.get("gsd_m_per_px"),
        "annotation_mode":     "sam_assisted",
        "hf_sync_status":      "pending",
        "hf_split":            None,
        "spatial_vector_data": None,
        "session_id":          None,
        "locked_by":           None,
        "locked_until":        None,
        "synced_to_hf":        False,
    }


def main() -> None:
    engine = create_engine(DB_URL, pool_pre_ping=True)

    sources = [
        ANNOTATIONS_DIR / "labeled.json",
        ANNOTATIONS_DIR / "unlabeled.json",
    ]

    rows: list[dict] = []
    seen_ids: set[str] = set()
    missing = 0

    for path in sources:
        if not path.exists():
            log.warning("Not found, skipping: %s", path)
            continue
        coco = load_coco(path)
        images = coco.get("images", [])
        log.info("Reading %s -> %d images", path.name, len(images))

        for img in images:
            row = build_row(img)
            if row is None:
                missing += 1
                continue
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            rows.append(row)

    if missing:
        log.warning("%d images not found on disk — skipped.", missing)

    log.info("Bulk inserting %d rows into Supabase...", len(rows))

    insert_sql = text("""
        INSERT INTO telemetry_components (
            id, file_path, confidence_index, matrix_class,
            is_baseline_anchor, validation_status,
            nac_id, patch_origin_x, patch_origin_y, gsd_m_per_px,
            annotation_mode, hf_sync_status, hf_split,
            spatial_vector_data, session_id, locked_by, locked_until,
            synced_to_hf
        ) VALUES (
            :id, :file_path, :confidence_index, :matrix_class,
            :is_baseline_anchor, :validation_status,
            :nac_id, :patch_origin_x, :patch_origin_y, :gsd_m_per_px,
            :annotation_mode, :hf_sync_status, :hf_split,
            :spatial_vector_data, :session_id, :locked_by, :locked_until,
            :synced_to_hf
        )
        ON CONFLICT (id) DO NOTHING
    """)

    CHUNK = 500
    total_inserted = 0

    with engine.begin() as conn:
        for i in range(0, len(rows), CHUNK):
            chunk = rows[i:i + CHUNK]
            result = conn.execute(insert_sql, chunk)
            total_inserted += result.rowcount
            log.info("  chunk %d/%d -> %d rows inserted",
                     i // CHUNK + 1, -(-len(rows) // CHUNK), result.rowcount)

    log.info("Done. total_inserted=%d  skipped(conflict)=%d",
             total_inserted, len(rows) - total_inserted)


if __name__ == "__main__":
    main()