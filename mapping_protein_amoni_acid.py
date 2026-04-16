import requests
import time
import logging
import json
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


active_ppi_path = "/DATA/DATANAS2/rhh25/Cellhit/data/graph_data/active_ppi_data.csv"

active_ppi_df = pd.read_csv(active_ppi_path)
active_ppi_df['protein1Clean'] = active_ppi_df['protein1'].str.split('.').str[1]
active_ppi_list = active_ppi_df['protein1Clean'].unique().tolist()




# ── Configuration ──────────────────────────────────────────────────────────────
ENSEMBL_SERVER   = "https://rest.ensembl.org"
BATCH_ENDPOINT   = "/sequence/id"
BATCH_SIZE       = 50
MAX_WORKERS      = 5
REQUESTS_PER_SEC = 10
RETRY_TOTAL      = 3
RETRY_BACKOFF    = 1.5
OUTPUT_JSON      = "protein_sequences.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Session with retry ─────────────────────────────────────────────────────────
def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


# ── Helpers ────────────────────────────────────────────────────────────────────
def _chunk(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _load_existing(path: str) -> dict[str, str | None]:
    """Load existing JSON file if it exists, so we can resume/update."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Core batch fetcher ─────────────────────────────────────────────────────────
_session = _build_session()
_min_interval = 1.0 / REQUESTS_PER_SEC
_last_request_time = 0.0


def _fetch_batch(protein_ids: list[str]) -> dict[str, str | None]:
    global _last_request_time

    elapsed = time.monotonic() - _last_request_time
    if elapsed < _min_interval:
        time.sleep(_min_interval - elapsed)
    _last_request_time = time.monotonic()

    headers = {
        "Content-Type": "application/json",
        "Accept":        "application/json",
    }
    payload = {"ids": protein_ids, "type": "protein"}
    results: dict[str, str | None] = {pid: None for pid in protein_ids}

    try:
        response = _session.post(
            ENSEMBL_SERVER + BATCH_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        for record in response.json():
            pid = record.get("id")
            seq = record.get("seq")
            if pid and seq:
                results[pid] = seq

    except requests.exceptions.HTTPError as e:
        log.error("HTTP error for batch %s: %s", protein_ids, e)
    except requests.exceptions.RequestException as e:
        log.error("Request error for batch %s: %s", protein_ids, e)
    except Exception as e:
        log.error("Unexpected error for batch %s: %s", protein_ids, e)

    return results


# ── Public API ─────────────────────────────────────────────────────────────────
def get_sequences_bulk(
    protein_ids: list[str],
    output_path: str = OUTPUT_JSON,
) -> dict[str, str | None]:
    """
    Fetches amino acid sequences for a large list of Ensembl protein IDs,
    updates an existing JSON file (resume-friendly), and returns the full dict.

    Args:
        protein_ids:  List of Ensembl protein IDs.
        output_path:  Path to the JSON file to update.

    Returns:
        Dict mapping each ID to its sequence string, or None if unavailable.
    """
    # Load existing results so previously fetched IDs are preserved
    all_results = _load_existing(output_path)

    # Only fetch IDs not already in the file
    ids_to_fetch = [pid for pid in protein_ids if pid not in all_results]

    if not ids_to_fetch:
        log.info("All %d IDs already present in %s. Nothing to fetch.",
                 len(protein_ids), output_path)
        return all_results

    batches = list(_chunk(ids_to_fetch, BATCH_SIZE))
    log.info(
        "Fetching %d new proteins (%d skipped) in %d batch(es) with %d workers …",
        len(ids_to_fetch), len(protein_ids) - len(ids_to_fetch),
        len(batches), MAX_WORKERS,
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(_fetch_batch, b): b for b in batches}

        with tqdm(total=len(ids_to_fetch), unit="protein", desc="Fetching") as pbar:
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.update(batch_results)
                    pbar.update(len(batch))

                    # Persist after every batch (crash-safe incremental writes)
                    _save_json(all_results, output_path)

                except Exception as e:
                    log.error("Batch %s raised an exception: %s", batch, e)
                    for pid in batch:
                        all_results[pid] = None
                    pbar.update(len(batch))

    log.info(
        "Done. %d/%d sequences saved to '%s'.",
        sum(v is not None for v in all_results.values()),
        len(all_results),
        output_path,
    )
    return all_results


# ── Main Execution ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    protein_ids = active_ppi_list

    sequences = get_sequences_bulk(protein_ids, output_path=OUTPUT_JSON)