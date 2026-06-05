---
title: Luna Labeler
emoji: 🌖
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Luna Labeler

Luna Labeler is a fast, specialized web-based tool designed for **instance segmentation labeling** on lunar surface images. It allows annotators to identify and outline features like pits, stones, and craters directly on a web interface using an overlay canvas. Label data is stored in a relational database (e.g., Supabase PostgreSQL) and can be exported as ML-ready segmentation masks.

## Key Features

- **Stateful Drawing Mode**: Supports drawing multiple polygons per image tile.
- **Multi-Class Support**: Distinct labels with custom visual coloring:
  - **`PIT`** (Green): Lunar pits and sinkholes.
  - **`STONE`** (Amber): Boulders and surface stones.
  - **`CRATER`** (Pink): Impact craters and depression zones.
- **Side-by-Side Interface**: A 512px canvas overlay on the right, with metadata, active tools, and action controls on the left.
- **Ergonomic Hotkeys**:
  - `1`, `2`, `3` to switch active tools.
  - `Space` to freeze/finish the current polygon.
  - `Z` / `z` to undo the last drawn node.
  - `Escape` to discard the current active polygon.
  - `Enter` to auto-freeze the current shape and submit the tile (Save & Next).
  - `Backspace` to undo/revert the last submitted annotation.
- **Binary Image Streaming**: Serves Hugging Face dataset images directly from memory cache via local API endpoints.
- **COCO Dataset Exporter**: Automatically formats and syncs verified segmentations into a standardized COCO Detection 2017 dataset on Hugging Face (`F1nnSBK/lunar-debris-and-voids`), using single-commit batch transactions to bypass rate limits.

---

## Project Structure

```
luna_labeler/
├── app/
│   ├── config.py           # Application settings and environment configurations
│   ├── database.py         # SQLAlchemy engine setup and DB session helpers
│   ├── main.py             # FastAPI backend routes & API endpoints (includes manual sync trigger)
│   ├── models.py           # Database tables/ORM models with COCO/provenance fields
│   ├── services/
│   │   └── telemetry_engine.py  # Logic for fetching and serving dataset anomalies
│   └── templates/
│       ├── index.html      # Main dashboard page structure
│       └── card_fragment.html   # Swappable HTMX card workspace (canvas, controls, hotkeys)
├── scripts/
│   ├── sync_hf_to_db.py    # Fetches HF dataset metadata and populates database
│   ├── migrate_coco_db.py  # Supabase DB schema migration script for COCO fields
│   ├── test_sync.py        # Helper to execute/test dataset synchronization locally
│   ├── populate_supabase.py# DB initialization and data population script
│   └── export_masks.py     # Renders DB annotations to raw/visual mask PNG files
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## Installation & Setup

1. **Clone the repository** and navigate to the project directory.

2. **Set up a Virtual Environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   TELEMETRY_DB_URL=postgresql://user:password@host:port/dbname
   TELEMETRY_TOKEN=your_huggingface_write_token
   ```

5. **Run DB migrations**:
   Apply schema changes to add COCO/provenance columns:
   ```bash
   uv run scripts/migrate_coco_db.py
   ```

6. **Initialize database schema and sync dataset**:
   Make sure to sync the dataset from Hugging Face to populate your database:
   ```bash
   uv run scripts/sync_hf_to_db.py
   ```

7. **Start the FastAPI Development Server**:
   ```bash
   uv run uvicorn app.main:app --reload
   ```
   Open your browser and navigate to `http://127.0.0.1:8000` to start labeling.

---

## Dataset Sync (COCO Format)

The system automatically pushes verified data to Hugging Face as a COCO Detection 2017 formatted dataset split into `train`, `val`, and `test` (using a 70/15/15 deterministic hashing split). Polygons are stored directly inside the annotation JSON files under `annotations/instances_{split}.json`, while images are located under `images/{split}/`.

To manually trigger a sync:
- Click the **FORCE SYNC TO HF** button on the UI dashboard stats panel, or
- Run the manual sync test script:
  ```bash
  uv run scripts/test_sync.py
  ```

