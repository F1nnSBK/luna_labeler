import base64
import random
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
    def resolve_next_payload(cls, execution_steps: int, db: Session) -> dict:
        # Variable ratio schedule (8% chance for a known true pit injection)
        if execution_steps > 0 and random.random() < 0.08:
            anchor = db.query(TelemetryComponent).filter_by(is_baseline_anchor=True).first()
            if anchor:
                return cls._build_response(anchor, "ANCHOR_HIT", "STOCHASTIC_ANCHOR")

        # Fallback to standard pending Spark anomalies
        item = db.query(TelemetryComponent).filter_by(validation_status="PENDING").first()
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