import asyncio
import io
import logging
import numpy as np
import cv2
from PIL import Image
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import TelemetryComponent

logger = logging.getLogger("active_learning")
logger.setLevel(logging.INFO)

# Global dataset cache reference initialized from main application on startup
hf_source_cache = None

def calculate_local_statistical_uncertainty(image_bytes: bytes) -> float:
    """
    Local statistical uncertainty based on edge complexity (variance of Laplacian)
    to prioritize complex structures (rocks, crater rims) without calling any external APIs.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.5
        
        # Heuristic: High variance of Laplacian indicates sharp edges and complexity.
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # Map variance (0 to 1000+) to range [0.0, 1.0] where 1.0 is high detail / high priority
        score = float(1.0 - np.exp(-var / 400.0))
        return score
    except Exception as e:
        logger.warning(f"Statistical complexity calculation failed: {e}")
        return 0.5

def score_unlabeled_pool(db: Session):
    """
    Selects 200 random PENDING components that are not locked,
    calculates their statistical complexity locally, and updates their confidence_index.
    """
    if hf_source_cache is None:
        logger.warning("hf_source_cache is not initialized. Skipping pool scoring.")
        return

    import datetime
    from sqlalchemy import or_
    now = datetime.datetime.now(datetime.timezone.utc)
    
    components = db.query(TelemetryComponent).filter(
        TelemetryComponent.validation_status == "PENDING",
        or_(
            TelemetryComponent.locked_until == None,
            TelemetryComponent.locked_until <= now
        )
    ).order_by(TelemetryComponent.confidence_index.asc()).limit(200).all()

    if not components:
        logger.info("No pending components to score.")
        return

    logger.info(f"Starting active learning statistical scoring loop for {len(components)} components...")
    
    scored_count = 0
    for item in components:
        try:
            split_name, row_idx = item.file_path.split("::")
            orig_img = hf_source_cache[split_name][int(row_idx)]["image"]
            
            img_buf = io.BytesIO()
            orig_img.save(img_buf, format="PNG")
            img_bytes = img_buf.getvalue()
            
            # Compute statistical uncertainty locally
            uncertainty = calculate_local_statistical_uncertainty(img_bytes)
            
            # High uncertainty (1.0) -> low confidence_index (0.0)
            item.confidence_index = float(1.0 - uncertainty)
            scored_count += 1
            
        except Exception as e:
            logger.error(f"Error scoring component {item.id}: {e}")
            
    try:
        db.commit()
        logger.info(f"Scored {scored_count} components using local statistical complexity.")
    except Exception as commit_err:
        db.rollback()
        logger.error(f"Failed to commit scored confidence indices: {commit_err}")

async def active_learning_loop():
    """Infinite loop scoring the unlabeled pool every 2 hours."""
    logger.info("Active learning background task worker started.")
    # Wait a bit on startup for dataset caching to populate
    await asyncio.sleep(15)
    
    while True:
        db = SessionLocal()
        try:
            score_unlabeled_pool(db)
        except Exception as e:
            logger.error(f"Active learning scoring loop encountered an error: {e}")
        finally:
            db.close()
            
        await asyncio.sleep(7200) # 2 hours
