from datasets import load_dataset
import os

token = os.getenv("TELEMETRY_TOKEN")
ds = load_dataset("F1nnSBK/lunar-pits-dataset", token=token)
for split in ds.keys():
    print(f"\n--- Split: {split} ---")
    print(f"Features: {ds[split].features}")
    print(f"Total rows: {len(ds[split])}")
    print("Row 0 keys & sample values:")
    item = ds[split][0]
    for k, v in item.items():
        if k == "image":
            print(f"  image: {type(v)}, filename: {getattr(v, 'filename', 'None')}")
        else:
            print(f"  {k}: {v}")
            
    # Print a few filenames from the image attribute if available
    print("First 10 image filenames:")
    for i in range(min(10, len(ds[split]))):
        img = ds[split][i]["image"]
        print(f"  idx {i}: label {ds[split][i].get('label')}, filename: {getattr(img, 'filename', 'None')}")
