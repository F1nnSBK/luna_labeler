"""
app/cron.py

Responsibilities:
  1. start_background_tasks() — starts active learning scoring loop (2h interval)
  2. get_db_stats()           — DB counters for the stats panel in the UI

COCO helpers (assign_split, polygon_to_bbox_area, validate_coco) are also
defined here and imported by scripts/build_hf_dataset.py.

HF sync is intentionally offline-only — use scripts/build_hf_dataset.py locally.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import TelemetryComponent

log = logging.getLogger(__name__)


# ── COCO helpers ──────────────────────────────────────────────────────────────

def assign_split(image_id_str: str, seed: str = "luna_v2") -> str:
    """Deterministically assigns train / val / test from image ID string."""
    h = int(hashlib.md5(f"{seed}_{image_id_str}".encode()).hexdigest(), 16)
    r = h % 100
    if r < 70:
        return "train"
    elif r < 85:
        return "val"
    return "test"


def polygon_to_bbox_area(coords: list[float]) -> tuple[list[float], float]:
    """
    coords: flat [x1, y1, x2, y2, ...] COCO polygon.
    Returns (bbox [x, y, w, h], area).
    """
    pts = np.array(coords).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox = [float(x_min), float(y_min),
            float(x_max - x_min), float(y_max - y_min)]
    x, y = pts[:, 0], pts[:, 1]
    area = float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return bbox, area


def validate_coco(coco: dict) -> None:
    """Raises AssertionError on any COCO structural violation."""
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


# ── DB stats ──────────────────────────────────────────────────────────────────

def get_db_stats(db: Session) -> dict[str, Any]:
    total    = db.query(TelemetryComponent).count()
    pending  = db.query(TelemetryComponent).filter_by(validation_status="PENDING").count()
    verified = db.query(TelemetryComponent).filter_by(validation_status="VERIFIED").count()
    synced   = db.query(TelemetryComponent).filter_by(hf_sync_status="synced").count()
    return {"total": total, "pending": pending, "verified": verified, "synced": synced}


# ── Background tasks ──────────────────────────────────────────────────────────

def start_background_tasks() -> None:
    """Schedule background loops. Call once from FastAPI @app.on_event('startup')."""
    from app.active_learning import active_learning_loop
    loop = asyncio.get_event_loop()
    loop.create_task(active_learning_loop())
    log.info("[cron] Active learning loop scheduled (interval: 2h).")