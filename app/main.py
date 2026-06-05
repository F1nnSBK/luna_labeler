"""
app/main.py

Luna Labeler — FastAPI application.
Images are served from local disk (data/active_learning_ds/images/).
HF sync is done offline via scripts/build_hf_dataset.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session
from ultralytics import SAM

from app.config import settings
from app.cron import get_db_stats, start_background_tasks
from app.database import Base, engine, get_db
from app.models import TelemetryComponent
from app.services.telemetry_engine import StochasticCalibrationEngine

# ── Bootstrap ─────────────────────────────────────────────────────────────────

Base.metadata.create_all(bind=engine)

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "active_learning_ds" / "images"

# MobileSAM weights — redirect to /tmp on HF Spaces (read-only fs)
if os.environ.get("HOME") == "/home/user" or not os.access(".", os.W_OK):
    weights_dir = Path("/tmp/weights")
else:
    weights_dir = Path(__file__).resolve().parent.parent / "weights"

weights_path = weights_dir / "mobile_sam.pt"
if not weights_path.exists():
    weights_dir.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt"
    print("Downloading MobileSAM weights...")
    urllib.request.urlretrieve(url, str(weights_path))
    print("MobileSAM weights ready.")

sam_model = SAM(str(weights_path))

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Luna Labeler")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    start_background_tasks()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serve_image(file_path: str) -> Response:
    """Serve an image from IMAGES_DIR. Raises 404 if not found."""
    local = IMAGES_DIR / file_path
    if not local.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {file_path}")
    return Response(content=local.read_bytes(), media_type="image/png")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def render_dashboard(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    session_id = request.cookies.get("labeler_session_id") or str(uuid.uuid4())[:12]
    payload = StochasticCalibrationEngine.resolve_next_payload(0, db, session_id)

    rendered = templates.TemplateResponse(
        "index.html",
        {"request": request, "payload": payload, "steps": 0,
         "stats": get_db_stats(db), "sync_result": None},
    )
    rendered.set_cookie("labeler_session_id", session_id, max_age=2592000)
    return rendered


@app.post("/dashboard/submit/{component_id}", response_class=HTMLResponse)
async def handle_submit(
    request: Request,
    component_id: str,
    execution_steps: int,
    spatial_vector_data: str = Form(default="[]"),
    db: Session = Depends(get_db),
):
    session_id = request.cookies.get("labeler_session_id", "anonymous")

    item = db.query(TelemetryComponent).filter_by(id=component_id).first()
    if item:
        item.spatial_vector_data = spatial_vector_data
        item.validation_status   = "VERIFIED"
        item.session_id          = session_id
        try:
            polygons = json.loads(spatial_vector_data)
            item.matrix_class = polygons[0].get("class", "UNKNOWN") if polygons else "EMPTY"
        except (json.JSONDecodeError, IndexError):
            item.matrix_class = "ERROR"
        db.commit()

    next_steps  = execution_steps + 1
    next_payload = StochasticCalibrationEngine.resolve_next_payload(next_steps, db, session_id)

    return templates.TemplateResponse(
        "card_fragment.html",
        {"request": request, "payload": next_payload, "steps": next_steps,
         "stats": get_db_stats(db), "sync_result": None},
    )


@app.post("/dashboard/undo", response_class=HTMLResponse)
async def handle_undo(
    request: Request,
    execution_steps: int,
    db: Session = Depends(get_db),
):
    session_id = request.cookies.get("labeler_session_id", "anonymous")

    last = (
        db.query(TelemetryComponent)
        .filter_by(validation_status="VERIFIED", session_id=session_id)
        .order_by(desc(TelemetryComponent.updated_at))
        .first()
    )

    if last:
        last.validation_status  = "PENDING"
        last.matrix_class       = "UNKNOWN"
        last.spatial_vector_data = None
        last.locked_by          = session_id
        last.locked_until       = datetime.now(timezone.utc) + timedelta(minutes=5)
        db.commit()
        payload = {
            "component_id":      last.id,
            "image_routing_url": f"/api/v1/image/{last.id}",
            "telemetry_string":  "UNDO — re-evaluate this patch.",
            "eval_tier":         "REVERTED",
        }
    else:
        payload = StochasticCalibrationEngine.resolve_next_payload(execution_steps, db, session_id)

    return templates.TemplateResponse(
        "card_fragment.html",
        {"request": request, "payload": payload,
         "steps": max(0, execution_steps - 1),
         "stats": get_db_stats(db), "sync_result": None},
    )


# ── Image serving ─────────────────────────────────────────────────────────────

@app.get("/api/v1/image/{component_id}")
async def stream_image(component_id: str, db: Session = Depends(get_db)):
    item = db.query(TelemetryComponent).filter_by(id=component_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Component not found")
    return _serve_image(item.file_path)


# ── SAM inference ─────────────────────────────────────────────────────────────

@app.post("/api/v1/sam/predict")
async def sam_predict(
    component_id: str = Form(...),
    x_min: float = Form(...),
    y_min: float = Form(...),
    x_max: float = Form(...),
    y_max: float = Form(...),
    db: Session = Depends(get_db),
):
    item = db.query(TelemetryComponent).filter_by(id=component_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Component not found")

    local = IMAGES_DIR / item.file_path
    if not local.exists():
        raise HTTPException(status_code=404, detail=f"Image not on disk: {item.file_path}")

    from PIL import Image as PILImage
    image_obj = PILImage.open(local).convert("RGB")

    try:
        results = sam_model.predict(image_obj, bboxes=[x_min, y_min, x_max, y_max], verbose=False)
        if results and results[0].masks is not None:
            xy = results[0].masks.xy
            if xy and len(xy[0]) > 0:
                return [[round(p[0]), round(p[1])] for p in xy[0].tolist()]
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM failed: {e}")


# ── Stats panel / manual sync trigger ────────────────────────────────────────

# Ersetze den /api/v1/sync/trigger Endpoint in main.py mit diesem:

@app.post("/api/v1/sync/trigger", response_class=HTMLResponse)
async def trigger_sync(request: Request, db: Session = Depends(get_db)):
    """
    Runs the full HF sync in-process:
    reads VERIFIED rows, builds COCO JSONs, uploads in one commit.
    """
    import asyncio
    from functools import partial

    sync_result = {"synced": 0, "error": None}

    try:
        # Run blocking sync in threadpool so we don't block the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_hf_sync)
        sync_result["synced"] = result
    except Exception as e:
        sync_result["error"] = str(e)[:80]

    stats = get_db_stats(db)
    return templates.TemplateResponse(
        "stats_panel.html",
        {"request": request, "stats": stats, "sync_result": sync_result},
    )


def _run_hf_sync() -> int:
    """
    Synchronous wrapper — runs the same logic as build_hf_dataset.py.
    Returns number of images synced.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Import build logic from the script
    from scripts.build_hf_dataset import (
        fetch_verified_rows,
        build_coco_splits,
        upload_to_hf,
        persist_splits,
    )
    from sqlalchemy import create_engine
    from app.config import settings

    engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    rows = fetch_verified_rows(engine)
    if not rows:
        return 0

    train, val, test, id_split_pairs = build_coco_splits(rows)
    upload_to_hf(train, val, test, rows)
    persist_splits(engine, id_split_pairs)
    return len(rows)


@app.get("/api/v1/telemetry/next")
async def get_next_payload(
    request: Request,
    execution_steps: int,
    db: Session = Depends(get_db),
):
    session_id = request.cookies.get("labeler_session_id", "api_worker")
    payload = StochasticCalibrationEngine.resolve_next_payload(execution_steps, db, session_id)
    if not payload:
        raise HTTPException(status_code=404, detail="No payloads available")
    return payload


class MetricPayload(BaseModel):
    assigned_label: str | None = None
    execution_steps: int
    session_id: str | None = None
    spatial_vector_data: str | None = None


@app.post("/api/v1/telemetry/submit/{component_id}")
async def submit_telemetry(
    component_id: str,
    payload: MetricPayload,
    db: Session = Depends(get_db),
):
    item = db.query(TelemetryComponent).filter_by(id=component_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Component not found")

    label = payload.assigned_label
    if not label and payload.spatial_vector_data:
        try:
            polys = json.loads(payload.spatial_vector_data)
            classes = {p.get("class") for p in polys if p.get("class")}
            label = next((c for c in ("PIT", "STONE", "CRATER") if c in classes), "UNKNOWN")
        except Exception:
            label = "UNKNOWN"

    item.matrix_class       = label or "UNKNOWN"
    item.validation_status  = "VERIFIED"
    item.session_id         = payload.session_id or "api_worker"
    if payload.spatial_vector_data:
        item.spatial_vector_data = payload.spatial_vector_data
    db.commit()

    return {"status": "SUCCESS", "logged_steps": payload.execution_steps + 1}