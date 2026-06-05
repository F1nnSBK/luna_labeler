import json
import csv
import re
from pathlib import Path

# Load catalogs
lpa_path = Path("/Users/finnhertsch/projects/luna_labeler/catalogs/lpa.csv")
pit_nacs_path = Path("/Users/finnhertsch/projects/luna_labeler/catalogs/pit_nacs.json")

# Read lpa.csv
pit_names = {}
with open(lpa_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pit_names[row['id']] = row['name']

# Read pit_nacs.json
with open(pit_nacs_path, 'r', encoding='utf-8') as f:
    pit_nacs = json.load(f)

# Build a mapping from base product ID (digits) to pit name
product_to_pit = {}
for pit_id, products in pit_nacs.items():
    name = pit_names.get(pit_id)
    if not name:
        continue
    for prod in products:
        prod_name = prod['product'] # e.g. "M1149067652R"
        # Extract base digits or exact product
        product_to_pit[prod_name] = name
        # Also map without L/R to be safe
        base_match = re.search(r'\d+', prod_name)
        if base_match:
            product_to_pit[base_match.group(0)] = name

print(f"Mapped {len(product_to_pit)} product keys to pit names.")

# Now let's try to match a few nac_ids
test_nacs = ["M106088433LC", "M1188488085LC", "M1149067652RC", "M108278624RC"]
for nac in test_nacs:
    # 1. Try exact product name match (stripped of trailing C/E)
    prod_stripped = nac
    if prod_stripped.endswith('C') or prod_stripped.endswith('E'):
        prod_stripped = prod_stripped[:-1]
    
    match = product_to_pit.get(prod_stripped)
    if not match:
        # Try base digits match
        digits_match = re.search(r'\d+', nac)
        if digits_match:
            match = product_to_pit.get(digits_match.group(0))
            
    print(f"NAC: {nac} -> Stripped: {prod_stripped} -> Pit Name: {match}")
