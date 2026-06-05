"""
app/active_learning.py

Scores PENDING patches by edge complexity (variance of Laplacian).
High complexity → low confidence_index → shown first in the labeler UI.
Runs every 2 hours as a background task started from app/cron.py.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import TelemetryComponent

logger = logging.getLogger("active_learning")

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "active_learning_ds" / "images"


def _laplacian_variance(img_path: Path) -> float:
    """Returns variance of Laplacian for a grayscale image. Higher = more edges."""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception as e:
        logger.warning("Could not score %s: %s", img_path.name, e)
        return 0.0


def score_unlabeled_pool(db: Session, batch_size: int = 200) -> int:
    """
    Scores up to batch_size unlocked PENDING components and updates confidence_index.
    Returns number of components scored.
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    components = (
        db.query(TelemetryComponent)
        .filter(
            TelemetryComponent.validation_status == "PENDING",
            or_(
                TelemetryComponent.locked_until == None,
                TelemetryComponent.locked_until <= now,
            ),
        )
        .order_by(TelemetryComponent.confidence_index.asc())
        .limit(batch_size)
        .all()
    )

    if not components:
        logger.info("No pending components to score.")
        return 0

    logger.info("Scoring %d components...", len(components))
    scored = 0

    for item in components:
        img_path = IMAGES_DIR / item.file_path
        if not img_path.exists():
            logger.warning("Image not found on disk: %s", item.file_path)
            continue

        var = _laplacian_variance(img_path)
        # High variance → complex image → low confidence_index → labelled first
        item.confidence_index = float(1.0 - np.exp(-var / 400.0))
        scored += 1

    try:
        db.commit()
        logger.info("Scored %d components.", scored)
    except Exception as e:
        db.rollback()
        logger.error("Commit failed after scoring: %s", e)

    return scored


async def active_learning_loop() -> None:
    """Runs every 2 hours. Called from app/cron.py start_background_tasks()."""
    logger.info("Active learning loop started (interval: 2h).")
    await asyncio.sleep(15)  # short delay to let FastAPI finish startup

    while True:
        db = SessionLocal()
        try:
            score_unlabeled_pool(db)
        except Exception as e:
            logger.error("Scoring loop error: %s", e)
            db.rollback()
        finally:
            db.close()

        await asyncio.sleep(7200)