# ── BATCH QUERY FUNCTION ─────────────────────────────────────────────────────

# ── CONFIG ──────────────────────────────────────────────────────────────────
import json
import pandas as pd
import requests
import time
from tqdm import tqdm


BATCH     = 2           # STRING API handles up to ~2000, but 100 is safe
SLEEP     = 1         # seconds between batches (be polite to the API)
STRING_API = "https://string-db.org/api/json/get_string_ids"


with open("bindingdb_unique_proteins.json", "r") as f:
    unique_proteins = json.load(f)

unique_proteins_demo = list(unique_proteins)

def query_string_batch(names: list[str]) -> list[dict]:
    params = {
        "identifiers"    : "\r".join(names),
        #"species"        : species,
        "limit"          : 3,
        "echo_query"     : 1,
        "caller_identity": "my_mapping_script"
    }
    r = requests.post(STRING_API, data=params, timeout=30)
    r.raise_for_status()
    return r.json()

# ── RUN IN BATCHES WITH TQDM ─────────────────────────────────────────────────
results = []
batches = [unique_proteins_demo[i : i + BATCH] for i in range(0, len(unique_proteins_demo), BATCH)]

for batch in tqdm(batches, desc="Querying STRING API", unit="batch"):
    try:
        hits = query_string_batch(batch)
        results.extend(hits)
    except Exception as e:
        tqdm.write(f"⚠ Batch failed: {e}")
    time.sleep(SLEEP)

# ── PARSE RESULTS ────────────────────────────────────────────────────────────
mapping = {}
for hit in tqdm(results, desc="Parsing results", unit="hit"):
    query_name = hit.get("queryItem")
    string_id  = hit.get("stringId")
    pref_name  = hit.get("preferredName")
#
    if query_name:
        if query_name not in mapping:
            mapping[query_name] = []       # init as list
        mapping[query_name].append({
            "STRING_id"      : string_id,
            "preferred_name" : pref_name,
        })

# ── FLATTEN: keep all hits as separate rows ───────────────────────────────────
mapping_df = pd.DataFrame([
    {"Target Name": name, **hit}
    for name, hits in mapping.items()
    for hit in hits
])
mapping_df.columns = ["Target Name", "STRING_id", "STRING_preferred_name"]


mapping_df.to_csv("bindingdb_protein_mapping_results.csv", index=False)

print("\n✅ Mapping complete. Results saved to 'bindingdb_protein_mapping_results.csv'.")