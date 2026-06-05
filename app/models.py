from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, Text, DateTime, Integer
from app.database import Base

class TelemetryComponent(Base):
    __tablename__ = "telemetry_components"
    
    id = Column(String, primary_key=True, index=True)
    file_path = Column(String, nullable=False)
    confidence_index = Column(Float, default=0.0)
    matrix_class = Column(String, default="UNKNOWN")
    is_baseline_anchor = Column(Boolean, default=False)
    validation_status = Column(String, default="PENDING")
    session_id = Column(String, nullable=True)
    spatial_vector_data = Column(Text, nullable=True)
    locked_by = Column(String, nullable=True, default=None)
    locked_until = Column(DateTime(timezone=True), nullable=True, default=None)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    synced_to_hf = Column(Boolean, default=False)
    
    # New COCO & Provenance fields
    nac_id = Column(String(32), nullable=True)
    patch_origin_x = Column(Integer, nullable=True)
    patch_origin_y = Column(Integer, nullable=True)
    gsd_m_per_px = Column(Float, nullable=True)
    annotation_mode = Column(String(16), default="sam_assisted")
    hf_sync_status = Column(String(16), default="pending")
    hf_split = Column(String(8), nullable=True)