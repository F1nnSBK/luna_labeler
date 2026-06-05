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
- **ML Mask Exporter**: Renders saved polygons into both raw categorical ML masks (values 0–3) and RGB visualization images.

---

## Project Structure

```
luna_labeler/
├── app/
│   ├── config.py           # Application settings and environment configurations
│   ├── database.py         # SQLAlchemy engine setup and DB session helpers
│   ├── main.py             # FastAPI backend routes & API endpoints
│   ├── models.py           # Database tables/ORM models (SQLAlchemy)
│   ├── services/
│   │   └── telemetry_engine.py  # Logic for fetching and serving dataset anomalies
│   └── templates/
│       ├── index.html      # Main dashboard page structure
│       └── card_fragment.html   # Swappable HTMX card workspace (canvas, controls, hotkeys)
├── scripts/
│   ├── sync_hf_to_db.py    # Fetches HF dataset metadata and populates database
│   ├── build_hf_dataset.py # Generates/syncs HF dataset assets
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

5. **Initialize database schema and sync dataset**:
   Make sure to sync the dataset from Hugging Face to populate your database:
   ```bash
   uv run scripts/sync_hf_to_db.py
   ```

6. **Start the FastAPI Development Server**:
   ```bash
   uv run uvicorn app.main:app --reload
   ```
   Open your browser and navigate to `http://127.0.0.1:8000` to start labeling.

---

## Mask Exporting

Once you have labeled and verified image tiles, run the export script to generate segmentation masks:
```bash
uv run scripts/export_masks.py
```
This outputs:
- **Raw Masks** (`data/export/masks_raw/`): Categorical integer values (0: Background, 1: Pit, 2: Stone, 3: Crater) for model training.
- **Visual Masks** (`data/export/masks_visual/`): Colorful RGB rendering matching the UI for human verification.
