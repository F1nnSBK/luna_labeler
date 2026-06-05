import os
import json
from pathlib import Path
from PIL import Image, ImageDraw
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models import TelemetryComponent

load_dotenv()

DB_URL = os.getenv("TELEMETRY_DB_URL")
ENGINE = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=ENGINE)

OUTPUT_DIR_RAW = Path("./data/export/masks_raw")
OUTPUT_DIR_VIS = Path("./data/export/masks_visual")

CLASS_MAPPING = {
    "BACKGROUND": 0,
    "PIT": 1,
    "STONE": 2,
    "CRATER": 3
}

# RGB values matching the UI for human verification
COLOR_PALETTE = {
    0: (0, 0, 0),
    1: (16, 185, 129),
    2: (245, 158, 11),
    3: (236, 72, 153)
}

def render_masks(vector_data: list, width: int = 256, height: int = 256) -> tuple[Image.Image, Image.Image]:
    mask_raw = Image.new('L', (width, height), color=CLASS_MAPPING["BACKGROUND"])
    mask_vis = Image.new('RGB', (width, height), color=COLOR_PALETTE[0])
    
    draw_raw = ImageDraw.Draw(mask_raw)
    draw_vis = ImageDraw.Draw(mask_vis)

    for feature in vector_data:
        cls_name = feature.get("class", "BACKGROUND")
        points = feature.get("points", [])
        
        if len(points) < 3:
            continue

        polygon_tuples = [(p[0], p[1]) for p in points]
        fill_val = CLASS_MAPPING.get(cls_name, 0)
        fill_color = COLOR_PALETTE.get(fill_val, (0, 0, 0))
        
        # Raw mask for ML (0, 1, 2, 3), Visual mask for human review (RGB)
        draw_raw.polygon(polygon_tuples, outline=fill_val, fill=fill_val)
        draw_vis.polygon(polygon_tuples, outline=fill_color, fill=fill_color)

    return mask_raw, mask_vis

def export_verified_datasets():
    OUTPUT_DIR_RAW.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_VIS.mkdir(parents=True, exist_ok=True)
    db = SessionLocal()
    
    try:
        components = db.query(TelemetryComponent).filter(
            TelemetryComponent.validation_status == "VERIFIED",
            TelemetryComponent.spatial_vector_data.isnot(None)
        ).all()

        exported_count = 0
        for comp in components:
            try:
                vectors = json.loads(comp.spatial_vector_data)
                mask_raw, mask_vis = render_masks(vectors)
                
                mask_raw.save(OUTPUT_DIR_RAW / f"{comp.id}_mask.png")
                mask_vis.save(OUTPUT_DIR_VIS / f"{comp.id}_vis.png")
                exported_count += 1
                
            except json.JSONDecodeError:
                continue

        print(f"Exported {exported_count} segmentation masks to {OUTPUT_DIR_RAW} and {OUTPUT_DIR_VIS}")
        
    finally:
        db.close()

if __name__ == "__main__":
    export_verified_datasets()
