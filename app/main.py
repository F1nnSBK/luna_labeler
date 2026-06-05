import uuid
import json
from fastapi import FastAPI, HTTPException, Depends, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db, engine, Base
from app.models import TelemetryComponent
from app.services.telemetry_engine import StochasticCalibrationEngine
import io
from datasets import load_dataset
from app.config import settings
import urllib.request
from pathlib import Path
from ultralytics import SAM

# Initialize database schema
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Lunar Telemetry Validation Service")

# Load dataset into memory on startup (339MB is easily handled by HF Spaces/local RAM)
hf_dataset_cache = load_dataset("F1nnSBK/lunar-pits-dataset", token=settings.HF_TOKEN)

# Check and download MobileSAM weights on startup
weights_dir = Path("weights")
weights_path = weights_dir / "mobile_sam.pt"
if not weights_path.exists():
    weights_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading MobileSAM weights...")
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt"
    try:
        urllib.request.urlretrieve(url, str(weights_path))
        print("MobileSAM weights downloaded successfully.")
    except Exception as e:
        print(f"Failed to download weights using urlretrieve: {e}. Trying fallback...")
        with urllib.request.urlopen(url) as response, open(weights_path, 'wb') as out_file:
            out_file.write(response.read())
        print("MobileSAM weights downloaded successfully (fallback method).")

# Load MobileSAM model
sam_model = SAM(str(weights_path))

templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MetricPayload(BaseModel):
    assigned_label: str | None = None
    execution_steps: int
    session_id: str | None = None
    spatial_vector_data: str | None = None


@app.get("/")
async def root_redirect():
    """Redirects root traffic to the calibration dashboard."""
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def render_dashboard_shell(request: Request, response: Response, db: Session = Depends(get_db)):
    """Renders dashboard and provisions a tracking session cookie if missing."""
    session_id = request.cookies.get("labeler_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())[:12]
    
    payload = StochasticCalibrationEngine.resolve_next_payload(0, db)
    
    rendered_template = templates.TemplateResponse(
        "index.html", 
        {"request": request, "payload": payload, "steps": 0}
    )
    rendered_template.set_cookie(key="labeler_session_id", value=session_id, max_age=2592000) # 30 days
    return rendered_template


@app.post("/dashboard/submit/{component_id}", response_class=HTMLResponse)
async def handle_dashboard_interact(
    request: Request, 
    component_id: str, 
    execution_steps: int, 
    spatial_vector_data: str = Form(default="[]"),
    db: Session = Depends(get_db)
):
    session_id = request.cookies.get("labeler_session_id", "anonymous_troll")
    
    item = db.query(TelemetryComponent).filter_by(id=component_id).first()
    if item:
        item.spatial_vector_data = spatial_vector_data
        item.validation_status = "VERIFIED"
        item.session_id = session_id
        
        try:
            polygons = json.loads(spatial_vector_data)
            item.matrix_class = polygons[0].get("class", "UNKNOWN") if polygons else "EMPTY"
        except json.JSONDecodeError:
            item.matrix_class = "ERROR"
            
        db.commit()
        
    next_steps = execution_steps + 1
    next_payload = StochasticCalibrationEngine.resolve_next_payload(next_steps, db)
    
    return templates.TemplateResponse(
        "card_fragment.html", 
        {"request": request, "payload": next_payload, "steps": next_steps}
    )


@app.get("/api/v1/telemetry/next")
async def get_next_telemetry_payload(execution_steps: int, db: Session = Depends(get_db)):
    payload = StochasticCalibrationEngine.resolve_next_payload(execution_steps, db)
    if not payload:
        raise HTTPException(status_code=404, detail="No pipeline payloads available")
    return payload


@app.get("/api/v1/image/{component_id}")
async def stream_component_image(component_id: str, db: Session = Depends(get_db)):
    item = db.query(TelemetryComponent).filter_by(id=component_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Component not found")

    split_name, row_idx = item.file_path.split("::")
    row_idx = int(row_idx)

    try:
        image_obj = hf_dataset_cache[split_name][row_idx]["image"]
        
        img_byte_arr = io.BytesIO()
        image_obj.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Image rendering failed")


@app.post("/api/v1/telemetry/submit/{component_id}")
async def submit_telemetry_validation(component_id: str, payload: MetricPayload, db: Session = Depends(get_db)):
    item = db.query(TelemetryComponent).filter_by(id=component_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Target component not found")
        
    inferred_label = payload.assigned_label
    if not inferred_label and payload.spatial_vector_data:
        try:
            import json
            polys = json.loads(payload.spatial_vector_data)
            classes = {p.get("class") for p in polys if p.get("class")}
            if "PIT" in classes:
                inferred_label = "PIT"
            elif "STONE" in classes:
                inferred_label = "STONE"
            elif "CRATER" in classes:
                inferred_label = "CRATER"
            else:
                inferred_label = "UNKNOWN"
        except Exception:
            inferred_label = "UNKNOWN"
            
    if not inferred_label:
        inferred_label = "UNKNOWN"

    item.matrix_class = inferred_label
    item.validation_status = "VERIFIED"
    item.session_id = payload.session_id or "api_worker"
    if payload.spatial_vector_data:
        item.spatial_vector_data = payload.spatial_vector_data
    db.commit()
    
    return {"status": "SUCCESS", "logged_steps": payload.execution_steps + 1}


@app.post("/api/v1/sam/predict")
async def sam_predict(
    component_id: str = Form(...),
    x_min: float = Form(...),
    y_min: float = Form(...),
    x_max: float = Form(...),
    y_max: float = Form(...),
    db: Session = Depends(get_db)
):
    """Run MobileSAM model inference to auto-generate a mask based on a bounding box prompt."""
    item = db.query(TelemetryComponent).filter_by(id=component_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Component not found")

    split_name, row_idx = item.file_path.split("::")
    row_idx = int(row_idx)

    try:
        image_obj = hf_dataset_cache[split_name][row_idx]["image"]
        if image_obj.mode != "RGB":
            image_obj = image_obj.convert("RGB")
            
        # Run MobileSAM prediction with bounding box coordinates mapping to 256x256 image pixels
        # Ultralytics SAM expects coordinates in format [x_min, y_min, x_max, y_max]
        results = sam_model.predict(image_obj, bboxes=[x_min, y_min, x_max, y_max], verbose=False)
        
        if results and len(results) > 0 and results[0].masks is not None:
            xy_coords = results[0].masks.xy
            if len(xy_coords) > 0 and len(xy_coords[0]) > 0:
                points = xy_coords[0].tolist()
                rounded_points = [[round(p[0]), round(p[1])] for p in points]
                return rounded_points
                
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM prediction failed: {str(e)}")


@app.post("/dashboard/undo", response_class=HTMLResponse)
async def handle_dashboard_undo(request: Request, execution_steps: int, db: Session = Depends(get_db)):
    """Reverts the last annotation for the current session and re-serves the component."""
    session_id = request.cookies.get("labeler_session_id", "anonymous_troll")
    
    # Find the absolute last component verified by this user
    last_item = db.query(TelemetryComponent).filter_by(
        validation_status="VERIFIED", 
        session_id=session_id
    ).order_by(desc(TelemetryComponent.updated_at)).first()
    
    if last_item:
        last_item.validation_status = "PENDING"
        last_item.matrix_class = "UNKNOWN"
        last_item.spatial_vector_data = None
        db.commit()
        
        # Override the payload with the reverted item
        payload = {
            "component_id": last_item.id,
            "image_routing_url": f"/api/v1/image/{last_item.id}",
            "telemetry_string": "UNDO TRIGGERED. RE-EVALUATE.",
            "eval_tier": "REVERTED_ANOMALY"
        }
    else:
        # If no history exists, just fetch the normal next item
        payload = StochasticCalibrationEngine.resolve_next_payload(execution_steps, db)

    return templates.TemplateResponse(
        "card_fragment.html", 
        {"request": request, "payload": payload, "steps": max(0, execution_steps - 1)}
    )