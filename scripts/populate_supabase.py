import os
import sys
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure local app path is visible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models import TelemetryComponent

# Grab the connection string directly from your local environment or .env
SUPABASE_DB_URL = os.getenv("TELEMETRY_DB_URL")
if not SUPABASE_DB_URL:
    print("ERROR: TELEMETRY_DB_URL environment variable is missing.")
    sys.exit(1)

engine = create_engine(SUPABASE_DB_URL)
SessionLocal = sessionmaker(bind=engine)

def seed_lunar_anomalies(detected_tiles: list):
    """
    Pushes extracted anomalies into Supabase. 
    Expects a list of dicts: [{'path': '...', 'score': 0.92, 'class': 'PIT'}]
    """
    db = SessionLocal()
    try:
        print(f"Connecting to Supabase to inject {len(detected_tiles)} elements...")
        
        for idx, tile in enumerate(detected_tiles):
            # Camouflage naming convention to keep Schutera oblivious
            is_anchor = tile.get("is_known_pit", False)
            
            component = TelemetryComponent(
                id=str(uuid.uuid4())[:8],  # Clean, short 8-character hash for the UI
                file_path=tile["path"],
                confidence_index=tile["score"],
                matrix_class=tile["class"],
                is_baseline_anchor=is_anchor,
                validation_status="PENDING" if not is_anchor else "VERIFIED"
            )
            db.add(component)
            
        db.commit()
        print("Database synchronization successful. System live.")
    except Exception as e:
        db.rollback()
        print(f"Database sync failed: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    # Mock data layout representing your pipeline outputs before pushing
    mock_pipeline_output = [
        {"path": "M188557729LC_patch_01.png", "score": 0.993, "class": "PIT", "is_known_pit": True}, # Honeypot
        {"path": "M188557729LC_patch_42.png", "score": 0.420, "class": "STONE", "is_known_pit": False},
        {"path": "M188557729LC_patch_73.png", "score": 0.510, "class": "CRATER", "is_known_pit": False}
    ]
    seed_lunar_anomalies(mock_pipeline_output)