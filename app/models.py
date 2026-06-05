from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, Text, DateTime
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
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)