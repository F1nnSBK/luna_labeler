import os
from pathlib import Path

ds_dir = Path("/Users/finnhertsch/projects/luna_labeler/data/active_learning_ds/images")
if ds_dir.exists():
    files = sorted(os.listdir(ds_dir))
    print(f"Total compiled images: {len(files)}")
    pos = [f for f in files if f.startswith("pos_")]
    neg = [f for f in files if f.startswith("neg_")]
    pot = [f for f in files if f.startswith("potential_")]
    print(f"pos: {len(pos)}, neg: {len(neg)}, potential: {len(pot)}")
    print("\nFirst 10 pos:")
    for f in pos[:10]:
        print(f)
    print("\nFirst 10 neg:")
    for f in neg[:10]:
        print(f)
    print("\nFirst 10 potential:")
    for f in pot[:10]:
        print(f)
else:
    print("Dataset directory does not exist")
