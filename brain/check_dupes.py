from datasets import load_dataset
import os
import re
from pathlib import Path

# Parse provenance function
def parse_provenance(path_str: str) -> tuple[str, str]:
    if not path_str or path_str == "None":
        return "UNKNOWN", ""
    filename = Path(path_str).name
    name, _ = os.path.splitext(filename)
    match = re.search(r'(M\d+[LR][CE])', name)
    if match:
        nac_id = match.group(1)
        prefix = name.split(nac_id)[0].rstrip("_")
        if prefix.startswith("neg_") or prefix == "neg":
            return nac_id, ""
        return nac_id, prefix
    return "UNKNOWN", ""

token = os.getenv("TELEMETRY_TOKEN")
ds = load_dataset("F1nnSBK/lunar-pits-dataset", token=token)

seen = set()
duplicates = []

for split in ds.keys():
    for idx, item in enumerate(ds[split]):
        if item.get("label") == 1:
            img = item["image"]
            filename = getattr(img, "filename", "")
            nac_id, pit_name = parse_provenance(filename)
            key = (nac_id, pit_name)
            if key in seen:
                duplicates.append((key, filename))
            seen.add(key)

print(f"Total positives: {len(seen) + len(duplicates)}")
print(f"Duplicate keys: {len(duplicates)}")
for d in duplicates:
    print(d)
