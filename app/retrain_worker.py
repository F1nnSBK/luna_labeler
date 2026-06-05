import asyncio
import io
import json
import logging
import os
from pathlib import Path
from sqlalchemy.orm import Session
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sqlalchemy import or_
import numpy as np

from app.database import SessionLocal
from app.models import TelemetryComponent
from app.cron import render_mask
from app.config import settings

logger = logging.getLogger("retrain_worker")
logger.setLevel(logging.INFO)

# Global dataset cache reference initialized from main application on startup
hf_source_cache = None

def get_bounding_box_from_mask(mask_np):
    """Computes bounding box [x_min, y_min, x_max, y_max] from binary mask."""
    y_indices, x_indices = np.where(mask_np > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0, 0, 256, 256]
    return [
        float(np.min(x_indices)),
        float(np.min(y_indices)),
        float(np.max(x_indices)),
        float(np.max(y_indices))
    ]

def train_sam_locally(db: Session, weights_path: Path):
    """
    Retrieves the last 50 verified telemetry components,
    fine-tunes the local MobileSAM prompt encoder & mask decoder,
    and updates the weights_path checkpoint.
    """
    if hf_source_cache is None:
        logger.warning("hf_source_cache is not initialized. Skipping retraining.")
        return

    # Fetch the 50 most recently verified items
    items = db.query(TelemetryComponent).filter_by(
        validation_status="VERIFIED"
    ).order_by(TelemetryComponent.updated_at.desc()).limit(50).all()

    if len(items) < 10:
        logger.info(f"Not enough verified items for training (found {len(items)}, need at least 10).")
        return

    logger.info("Initializing local MobileSAM training loop...")

    # Load MobileSAM using ultralytics and extract the underlying PyTorch model
    from ultralytics import SAM
    sam_model = SAM(str(weights_path))
    model = sam_model.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze Image Encoder
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # Ensure Prompt Encoder and Mask Decoder parameters are trainable
    for param in model.prompt_encoder.parameters():
        param.requires_grad = True
    for param in model.mask_decoder.parameters():
        param.requires_grad = True

    # Optimizer (learning rate 1e-5 is standard for fine-tuning to prevent catastrophic forgetting)
    optimizer = optim.Adam(
        list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()),
        lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    epochs = 8
    batch_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for item in items:
            try:
                split_name, row_idx = item.file_path.split("::")
                orig_img = hf_source_cache[split_name][int(row_idx)]["image"]
                if orig_img.mode != "RGB":
                    orig_img = orig_img.convert("RGB")

                # 1. Render mask (256x256 target mask)
                mask_bytes = render_mask(item.spatial_vector_data)
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
                mask_np = np.array(mask_img) / 255.0

                # 2. Get bounding box prompt from the mask
                bbox = get_bounding_box_from_mask(mask_np)

                # Preprocess image to SAM inputs (standard size 1024x1024)
                img_resized = orig_img.resize((1024, 1024))
                img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float().to(device)
                # Normalize (SAM expect inputs normalized, approximate here)
                img_tensor = (img_tensor - 127.5) / 127.5
                img_tensor = img_tensor.unsqueeze(0) # [1, 3, 1024, 1024]

                # Target mask resized to 256x256 (SAM decoder output size)
                mask_resized = mask_img.resize((256, 256))
                target_mask_tensor = torch.from_numpy(np.array(mask_resized) / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)

                # Bounding box prompt format: [[x_min, y_min, x_max, y_max]] mapped to 1024x1024
                # Since mask_np was 256x256, scale bbox coordinates to 1024x1024
                scale_x = 1024.0 / 256.0
                scale_y = 1024.0 / 256.0
                scaled_bbox = [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y
                ]
                
                # Format boxes for prompt encoder
                box_tensor = torch.tensor([scaled_bbox], dtype=torch.float, device=device).unsqueeze(0) # [1, 1, 4]

                optimizer.zero_grad()

                # Forward Pass
                # Get image embeddings
                image_embeddings = model.image_encoder(img_tensor)

                # Encode Prompts
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=box_tensor,
                    masks=None
                )

                # Decode Masks
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                loss = criterion(low_res_masks, target_mask_tensor)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            except Exception as e:
                logger.error(f"Error training on item {item.id}: {e}")

        logger.info(f"Epoch {epoch+1}/{epochs} complete. Average Loss: {epoch_loss / len(items):.4f}")

    # Save weights back safely
    try:
        checkpoint = torch.load(weights_path, map_location="cpu")
        checkpoint["model"] = model.to("cpu").state_dict()
        torch.save(checkpoint, weights_path)
        logger.info("MobileSAM weights successfully updated and saved locally.")
    except Exception as save_err:
        logger.error(f"Failed to save trained weights: {save_err}")

async def retrain_worker_loop(weights_path: Path):
    """
    Monitors verified database status delta and triggers
    training loop after every 50 verified items.
    """
    logger.info("Retrain background worker started.")
    last_trained_count = 0
    
    # Wait a bit on startup
    await asyncio.sleep(20)

    while True:
        db = SessionLocal()
        try:
            # Check number of verified items
            current_verified_count = db.query(TelemetryComponent).filter_by(
                validation_status="VERIFIED"
            ).count()

            # Initialize last_trained_count on first run
            if last_trained_count == 0:
                last_trained_count = current_verified_count

            delta = current_verified_count - last_trained_count
            logger.info(f"Retrain check: {current_verified_count} verified. Delta since last train: {delta}/50")

            if delta >= 50:
                train_sam_locally(db, weights_path)
                last_trained_count = current_verified_count

        except Exception as e:
            logger.error(f"Retrain worker loop encountered an error: {e}")
        finally:
            db.close()

        await asyncio.sleep(120) # Check every 2 minutes
