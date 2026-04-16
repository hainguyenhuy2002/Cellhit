import requests
import time
import logging
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────
data_path = Path('/DATA/DATANAS2/rhh25/Cellhit/data/')

df_drugcombs = pd.read_csv(f'{data_path}/metadata/drugcombs_scored.csv')
df_drugcombs = df_drugcombs.dropna(subset=['Drug1', 'Drug2'] )
unique_drugs =  set(df_drugcombs.Drug1.unique().tolist()).union(
        set(df_drugcombs.Drug2.unique().tolist())
    )

# with open("drug_smiles.json", "r") as f:
#     smiles_drugs = list(json.load(f).keys())

# with open("drug_smiles1.json", "r") as f:
#     smiles_drugs1 = list(json.load(f).keys())

# with open("drug_smiles2.json", "r") as f:
#     smiles_drugs2 = list(json.load(f).keys())

# with open("drug_not_found.json", "r") as f:
#     smiles_drugs_notfound = json.load(f)['not_found']

# with open("drug_not_found1.json", "r") as f:
#     smiles_drugs_notfound_1 = json.load(f)['not_found']

# with open("drug_not_found2.json", "r") as f:
#     smiles_drugs_notfound_2 = json.load(f)['not_found']

# remain_drugs = list(unique_drugs - set(smiles_drugs) - set(smiles_drugs1) -set(smiles_drugs2) - set(smiles_drugs_notfound) - set(smiles_drugs_notfound_1) - set(smiles_drugs_notfound_2))




PUBCHEM_SERVER      = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CID_BATCH_SIZE      = 16
MAX_WORKERS         = 1           # ↓ reduced: fewer parallel threads → fewer bursts
REQUESTS_PER_MIN    = 70         # PubChem guideline: ≤ 5 req/s = 300/min; stay at 100 to be safe
RETRY_TOTAL         = 4
RETRY_BACKOFF       = 2.0
OUTPUT_JSON         = "drug_smiles.json"
NOT_FOUND_JSON      = "drug_not_found.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Thread-safe rate limiter ───────────────────────────────────────────────────
class RateLimiter:
    """
    Token-bucket rate limiter — thread-safe.
    Ensures at most `max_calls` requests per `period` seconds across all threads.
    """
    def __init__(self, max_calls: int, period: float = 60.0):
        self._lock        = threading.Lock()
        self._max_calls   = max_calls
        self._period      = period
        self._min_interval = period / max_calls   # seconds between each call
        self._last_call   = 0.0

    def wait(self):
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last_call
            wait_time = self._min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_call = time.monotonic()


_rate_limiter = RateLimiter(max_calls=REQUESTS_PER_MIN, period=60.0)


# ── Session with retry + 429 back-off ─────────────────────────────────────────
def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True,   # honours PubChem's Retry-After header
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
    )
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


_session = _build_session()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _chunk(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _load_existing(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(data: dict | list, path: str) -> None:
    """Atomic write — prevents corrupt JSON on crash."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _handle_429(response: requests.Response) -> None:
    """Explicit sleep when a 429 slips through the retry adapter."""
    retry_after = int(response.headers.get("Retry-After", 10))
    log.warning("429 Too Many Requests — sleeping %ds …", retry_after)
    time.sleep(retry_after)


# ── Step 1: Name → CID ────────────────────────────────────────────────────────
def _name_to_cid_single(drug_name: str) -> tuple[str, int | None]:
    _rate_limiter.wait()                                   # ← thread-safe wait
    encoded = requests.utils.quote(drug_name, safe="")    # ← safe="" encodes '/'
    url = f"{PUBCHEM_SERVER}/compound/name/{encoded}/cids/JSON"
    try:
        r = _session.get(url, timeout=20)
        if r.status_code == 404:
            return drug_name, None
        if r.status_code == 429:
            _handle_429(r)
            return _name_to_cid_single(drug_name)          # one manual retry
        r.raise_for_status()
        cids = r.json().get("IdentifierList", {}).get("CID", [])
        return drug_name, cids[0] if cids else None
    except requests.exceptions.Timeout:
        log.warning("Timeout for '%s' — will be marked not found.", drug_name)
        return drug_name, None
    except Exception as e:
        log.error("CID lookup failed for '%s': %s", drug_name, e)
        return drug_name, None


# def resolve_names_to_cids(drug_names: list[str]) -> dict[str, int | None]:
#     name_to_cid: dict[str, int | None] = {}
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(_name_to_cid_single, n): n for n in drug_names}
#         with tqdm(total=len(drug_names), unit="drug", desc="Resolving names → CIDs") as pbar:
#             for future in as_completed(futures):
#                 name, cid = future.result()
#                 name_to_cid[name] = cid
#                 pbar.update(1)

#     resolved = sum(v is not None for v in name_to_cid.values())
#     log.info("Resolved %d/%d drug names to CIDs.", resolved, len(drug_names))
#     return name_to_cid

def resolve_names_to_cids(
    drug_names: list[str],
    pbar: tqdm,
) -> dict[str, int | None]:
    name_to_cid: dict[str, int | None] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_name_to_cid_single, n): n for n in drug_names}
        for future in as_completed(futures):
            name, cid = future.result()
            name_to_cid[name] = cid
            pbar.update(1)                     # ← tick once per drug

    resolved = sum(v is not None for v in name_to_cid.values())
    log.info("Resolved %d/%d drug names to CIDs.", resolved, len(drug_names))
    return name_to_cid

# ── Step 2: CID → SMILES ──────────────────────────────────────────────────────
def _fetch_smiles_batch(cid_batch: list[int]) -> dict[int, str | None]:
    _rate_limiter.wait()                                   # ← thread-safe wait
    cids_str = ",".join(map(str, cid_batch))
    url = f"{PUBCHEM_SERVER}/compound/cid/{cids_str}/property/ConnectivitySMILES/JSON"
    results = {cid: None for cid in cid_batch}
    try:
        r = _session.get(url, timeout=30)
        if r.status_code == 429:
            _handle_429(r)
            return _fetch_smiles_batch(cid_batch)          # one manual retry
        r.raise_for_status()
        for prop in r.json().get("PropertyTable", {}).get("Properties", []):
            results[prop["CID"]] = prop.get("ConnectivitySMILES")
    except requests.exceptions.Timeout:
        log.warning("Timeout for CID batch %s… — skipping.", cid_batch[:3])
    except requests.exceptions.HTTPError as e:
        log.error("SMILES batch fetch failed (CIDs: %s…): %s", cid_batch[:3], e)
    except Exception as e:
        log.error("Unexpected error fetching SMILES batch: %s", e)
    return results


# def fetch_smiles_for_cids(cid_list: list[int]) -> dict[int, str | None]:
#     cid_to_smiles: dict[int, str | None] = {}
#     batches = list(_chunk(cid_list, CID_BATCH_SIZE))
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(_fetch_smiles_batch, b): b for b in batches}
#         with tqdm(total=len(cid_list), unit="CID", desc="Fetching SMILES") as pbar:
#             for future in as_completed(futures):
#                 batch_result = future.result()
#                 cid_to_smiles.update(batch_result)
#                 pbar.update(len(futures[future]))
#     return cid_to_smiles


def fetch_smiles_for_cids(
    cid_list: list[int],                       # ← no pbar here anymore
) -> dict[int, str | None]:
    cid_to_smiles: dict[int, str | None] = {}
    batches = list(_chunk(cid_list, CID_BATCH_SIZE))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_smiles_batch, b): b for b in batches}
        for future in as_completed(futures):
            batch_result = future.result()
            cid_to_smiles.update(batch_result)
    return cid_to_smiles

def get_smiles_bulk(
    drug_names: list[str],
    output_path: str = OUTPUT_JSON,
    not_found_path: str = NOT_FOUND_JSON,
    checkpoint_size: int = 10,
) -> tuple[dict[str, dict | None], list[str]]:
    all_results       = _load_existing(output_path)
    not_found_data    = _load_existing(not_found_path)
    already_not_found = set(not_found_data.get("not_found", []))

    names_to_fetch = [
        n for n in drug_names
        if n not in all_results and n not in already_not_found
    ]

    if not names_to_fetch:
        log.info("All %d drugs already processed. Nothing to fetch.", len(drug_names))
        return all_results, list(already_not_found)

    log.info(
        "%d drugs to fetch | %d cached | %d previously not found.",
        len(names_to_fetch),
        len([n for n in drug_names if n in all_results]),
        len(already_not_found & set(drug_names)),
    )

    all_not_found = list(already_not_found)

    with tqdm(total=len(names_to_fetch), unit="drug", desc="Overall progress") as pbar:

        for chunk_start in range(0, len(names_to_fetch), checkpoint_size):
            chunk = names_to_fetch[chunk_start : chunk_start + checkpoint_size]

            # ── Step 1: names → CIDs (ticks pbar once per drug) ───────────────
            name_to_cid     = resolve_names_to_cids(chunk, pbar)
            newly_not_found = [name for name, cid in name_to_cid.items() if cid is None]
            all_not_found   = sorted(set(all_not_found) | set(newly_not_found))

            # ── Step 2: CIDs → SMILES ─────────────────────────────────────────
            valid_cids = list({cid for cid in name_to_cid.values() if cid is not None})

            cid_to_names: dict[int, list[str]] = defaultdict(list)
            for name, cid in name_to_cid.items():
                if cid is not None:
                    cid_to_names[cid].append(name)

            cid_to_smiles = fetch_smiles_for_cids(valid_cids)

            # ── Merge ─────────────────────────────────────────────────────────
            for cid, names in cid_to_names.items():
                smiles = cid_to_smiles.get(cid)
                for name in names:
                    all_results[name] = {"CID": cid, "ConnectivitySMILES": smiles}

            # ── Save every N drugs (once per checkpoint) ───────────────────────
            _save_json(all_results, output_path)
            _save_json(
                {
                    "not_found":    all_not_found,
                    "total":        len(all_not_found),
                    "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                not_found_path,
            )
            log.info(
                "Checkpoint saved — %d/%d drugs processed so far.",
                chunk_start + len(chunk), len(names_to_fetch),
            )

    found = sum(1 for v in all_results.values() if v and v.get("ConnectivitySMILES"))
    log.info(
        "Done. %d/%d drugs have SMILES → '%s' | %d not found → '%s'.",
        found, len(all_results), output_path, len(all_not_found), not_found_path,
    )
    return all_results, all_not_found

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, not_found = get_smiles_bulk(
        unique_drugs,
        output_path=OUTPUT_JSON,
        not_found_path=NOT_FOUND_JSON,
    )