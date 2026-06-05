import os
from dotenv import load_dotenv

# Automatically load environment variables from .env for local development
load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("TELEMETRY_DB_URL", "postgresql://user:pass@localhost:5432/luna")
    HF_TOKEN: str = os.getenv("TELEMETRY_TOKEN", "")
    HF_DATASET_ID: str = "F1nnSBK/lunar-telemetry-assets"

settings = Settings()