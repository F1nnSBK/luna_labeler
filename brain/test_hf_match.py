from datasets import load_dataset
import os
import json
import csv
import re
from pathlib import Path

# Load catalogs
lpa_path = Path("/Users/finnhertsch/projects/luna_labeler/catalogs/lpa.csv")
pit_nacs_path = Path("/Users/finnhertsch/projects/luna_labeler/catalogs/pit_nacs.json")

pit_names = {}
with open(lpa_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pit_names[row['id']] = row['name']

with open(pit_nacs_path, 'r', encoding='utf-8') as f:
    pit_nacs = json.load(f)

product_to_pit = {}
for pit_id, products in pit_nacs.items():
    name = pit_names.get(pit_id)
    if not name:
        continue
    for prod in products:
        prod_name = prod['product']
        product_to_pit[prod_name] = name
        # Map digits too
        digits = re.search(r'\d+', prod_name)
        if digits:
            product_to_pit[digits.group(0)] = name

def get_pit_name(nac_id: str) -> str:
    # Try exact product name match (stripped of trailing C/E)
    prod_stripped = nac_id
    if prod_stripped.endswith('C') or prod_stripped.endswith('E'):
        prod_stripped = prod_stripped[:-1]
    
    match = product_to_pit.get(prod_stripped)
    if not match:
        digits_match = re.search(r'\d+', nac_id)
        if digits_match:
            match = product_to_pit.get(digits_match.group(0))
    return match or ""

token = os.getenv("TELEMETRY_TOKEN")
ds = load_dataset("F1nnSBK/lunar-pits-dataset", token=token)

total_positives = 0
matched_names = 0
unmatched_nacs = set()

for split in ds.keys():
    for item in ds[split]:
        if item.get("label") == 1:
            total_positives += 1
            img = item["image"]
            filename = getattr(img, "filename", "")
            
            # Parse NAC ID
            name, _ = os.path.splitext(Path(filename).name)
            match = re.search(r'(M\d+[LR][CE])', name)
            if match:
                nac_id = match.group(1)
                resolved = get_pit_name(nac_id)
                if resolved:
                    matched_names += 1
                else:
                    unmatched_nacs.add(nac_id)
            else:
                print(f"Failed to parse NAC ID from: {filename}")

print(f"Total Positives: {total_positives}")
print(f"Matched Names: {matched_names}")
print(f"Unmatched unique NACs: {unmatched_nacs}")
