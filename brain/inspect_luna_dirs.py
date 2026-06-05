import os
from pathlib import Path

LUNA_ROOT = Path("/Users/finnhertsch/projects/luna")

dirs = [
    LUNA_ROOT / "data" / "vit_dataset" / "negatives",
    LUNA_ROOT / "data" / "vit_dataset" / "negatives_active",
    LUNA_ROOT / "data" / "vit_dataset" / "potentials",
    LUNA_ROOT / "temp"
]

for d in dirs:
    print(f"\n--- Directory: {d} ---")
    if d.exists():
        files = sorted(os.listdir(d))
        print(f"Total files: {len(files)}")
        for f in files[:10]:
            print(f)
    else:
        print("Does not exist")
