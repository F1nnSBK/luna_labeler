import base64
import random
from datetime import datetime, timedelta, timezone
from sqlalchemy import or_
from sqlalchemy.orm import Session
from app.models import TelemetryComponent
from app.config import settings

_TELEMETRY_SIGNALS = {
    "ANCHOR_HIT": ["U2FmZSBMb2NoLiBTaGVlc2gu", "Qm9kZW5sb3MgZGVlcCwgc2xheS4=", "V2lsZGVyIFNjaGx1bmQsIG5vIGNhcC4="],
    "BOULDER_MISSED": ["Q3JpbmdlLCB3ZXIgZGFzIGbDdXIgZWluIExvY2ggaMOkbHQu", "QnJ1ZGVyLCBkYXMgaXN0IGVpbiBTdGVpbi4gTmV4dC4="],
    "CRATER_BLENDER": ["R290dGxvc2VyIEtyYXRlciwgYWJzb2x1dGVyIEJsZW5kZXIu", "TnVyIFNjaGF0dGVuLCBsb3drZXkgTC4="]
}

class StochasticCalibrationEngine:
    @staticmethod
    def _decode_signal(category: str) -> str:
        return base64.b64decode(random.choice(_TELEMETRY_SIGNALS[category])).decode("utf-8")

    @classmethod
    def resolve_next_payload(cls, execution_steps: int, db: Session, session_id: str = None) -> dict:
        # Variable ratio schedule (8% chance for a known true pit injection)
        if execution_steps > 0 and random.random() < 0.08:
            anchor = db.query(TelemetryComponent).filter_by(is_baseline_anchor=True).first()
            if anchor:
                return cls._build_response(anchor, "ANCHOR_HIT", "STOCHASTIC_ANCHOR")

        now = datetime.now(timezone.utc)
        
        # 1. See if we already have an item locked by this user but not yet verified
        if session_id:
            item = db.query(TelemetryComponent).filter(
                TelemetryComponent.validation_status == "PENDING",
                TelemetryComponent.locked_by == session_id,
                TelemetryComponent.locked_until > now
            ).first()
        else:
            item = None

        if not item:
            # 2. Find a new one and lock it atomically
            item = db.query(TelemetryComponent).filter(
                TelemetryComponent.validation_status == "PENDING",
                or_(
                    TelemetryComponent.locked_until == None,
                    TelemetryComponent.locked_until <= now
                )
            ).order_by(TelemetryComponent.confidence_index.asc()).with_for_update(skip_locked=True).first()
            
            if item and session_id:
                item.locked_by = session_id
                item.locked_until = now + timedelta(minutes=5)
                db.commit()

        if not item:
            return {}

        category = "BOULDER_MISSED" if item.confidence_index < 0.50 and item.matrix_class == "STONE" else "CRATER_BLENDER"
        return cls._build_response(item, category, "STANDARD_ANOMALY")

    @classmethod
    def _build_response(cls, item: TelemetryComponent, category: str, tier: str) -> dict:
        return {
            "component_id": item.id,
            "image_routing_url": f"/api/v1/image/{item.id}",
            "telemetry_string": cls._decode_signal(category),
            "eval_tier": tier
        }