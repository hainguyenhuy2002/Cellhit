"""
Bulk SMILES → PubChem CID mapper
---------------------------------
- Threaded concurrency (concurrent.futures.ThreadPoolExecutor): zero blocking wait
- Token-bucket rate limiter (REQUESTS_PER_MIN hard cap) — lock released before sleep
- Exponential backoff on transient failures
- Checkpoint saving every N drugs processed
- tqdm progress bar with live retry/wait visibility
- Two output JSON files:
    - success_output.json : { smiles -> [cids] }
    - failed_output.json  : { smiles -> error_reason }
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from services.get_ood_data import get_ood_data

# ───────────────────────��──────────────────────
# Configuration
# ──────────────────────────────────────────────
SUCCESS_OUTPUT_FILE = "cid_smile_mapping.json"
FAILED_OUTPUT_FILE  = "failed_cid_smile_mapping.json"

BATCH_SIZE         = 16    # SMILES per batch submitted to the thread pool
CHECKPOINT_EVERY   = 100   # Save to disk every N drugs processed
MAX_WORKERS        = 1     # Max concurrent threads
REQUESTS_PER_MIN   = 100   # Hard cap on requests per minute (token-bucket)
MAX_RETRIES        = 5     # Retry attempts per request on failure
BACKOFF_BASE       = 2.0   # Exponential backoff base (seconds): 2, 4, 8
REQUEST_TIMEOUT    = 30    # Seconds before a request times out

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
)

# ──────────────────────────────────────────────
# Logging (file only — tqdm owns stdout)
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("smiles_to_cid.log")],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Token-bucket rate limiter (fixed: lock NOT held during sleep)
# ──────────────────────────────────────────────
class TokenBucketRateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Critical fix vs previous version:
        The lock is released BEFORE sleeping, so other threads are not
        blocked while one thread waits for its token. Each thread
        calculates its own required wait time under the lock, then
        releases it and sleeps independently.

    Args:
        rate (int): Maximum number of requests allowed per minute.
    """

    def __init__(self, rate: int) -> None:
        self._interval  = 60.0 / rate   # seconds between tokens
        self._lock      = threading.Lock()
        self._last_time = time.monotonic()

    def acquire(self) -> None:
        """Calculate wait time under lock, release lock, then sleep."""
        with self._lock:
            now  = time.monotonic()
            wait = self._last_time + self._interval - now
            # Reserve the next token slot regardless of whether we sleep
            self._last_time = max(now, self._last_time) + self._interval

        # Sleep OUTSIDE the lock — other threads can get their slots concurrently
        if wait > 0:
            time.sleep(wait)


# Single shared limiter used by all threads
_rate_limiter = TokenBucketRateLimiter(rate=REQUESTS_PER_MIN)


# ──────────────────────────────────────────────
# Per-thread requests Session
# ──────────────────────────────────────────────
_thread_local = threading.local()

def get_session() -> requests.Session:
    """Return a thread-local requests.Session (created once per thread)."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=Retry(total=0, raise_on_status=False),
            pool_connections=MAX_WORKERS,
            pool_maxsize=MAX_WORKERS,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return _thread_local.session


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────
def load_checkpoint(filepath: str) -> dict:
    """Load existing results from a JSON file (resume support)."""
    p = Path(filepath)
    if p.exists():
        with open(p, "r") as f:
            data = json.load(f)
        tqdm.write(f"[resume] Loaded {len(data)} existing entries from '{filepath}'")
        return data
    return {}


def save_checkpoint(data: dict, filepath: str) -> None:
    """Atomically write results to a JSON file via tmp → rename."""
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    Path(tmp).replace(filepath)
    log.info(f"Checkpoint saved → '{filepath}' ({len(data)} entries)")
    tqdm.write(f"[checkpoint] Saved '{filepath}' ({len(data)} entries)")


# ──────────────────────────────────────────────
# Core fetch (single SMILES, runs in a thread)
# ──────────────────────────────────────────────
def fetch_cids(smiles: str) -> tuple[str, list[int] | None, str | None]:
    """
    Fetch CID(s) for a single SMILES string with exponential backoff.
    Acquires a rate-limiter token before every HTTP request.

    Returns:
        (smiles, [cids], None)        on success
        (smiles, None, error_reason)  on failure
    """
    session = get_session()
    url = PUBCHEM_URL.format(smiles=quote(smiles, safe=""))

    for attempt in range(1, MAX_RETRIES + 1):
        _rate_limiter.acquire()  # ← non-blocking for other threads
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 200:
                cids = resp.json().get("IdentifierList", {}).get("CID", [])
                return smiles, cids, None

            elif resp.status_code == 404:
                # Definitive: compound not found — do not retry
                return smiles, None, "HTTP 404: compound not found"

            elif resp.status_code in (429, 503):
                wait = BACKOFF_BASE ** attempt  # 2s, 4s, 8s
                log.warning(
                    f"Rate limited (HTTP {resp.status_code}) for "
                    f"'{smiles[:30]}', attempt {attempt}/{MAX_RETRIES}, "
                    f"waiting {wait:.1f}s"
                )
                tqdm.write(
                    f"[backoff] HTTP {resp.status_code} → waiting {wait:.1f}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(wait)

            else:
                wait = BACKOFF_BASE ** attempt
                log.warning(
                    f"HTTP {resp.status_code} for '{smiles[:30]}', "
                    f"attempt {attempt}/{MAX_RETRIES}, waiting {wait:.1f}s"
                )
                time.sleep(wait)

        except requests.exceptions.RequestException as exc:
            wait = BACKOFF_BASE ** attempt
            log.warning(
                f"Network error for '{smiles[:30]}': {exc}, "
                f"attempt {attempt}/{MAX_RETRIES}, waiting {wait:.1f}s"
            )
            tqdm.write(f"[error] Network error ({exc}) → waiting {wait:.1f}s")
            time.sleep(wait)

    return smiles, None, f"Failed after {MAX_RETRIES} retries"


# ──────────────────────────────────────────────
# Batch processor with checkpointing + tqdm
# ──────────────────────────────────────────────
def process_all(smiles_list: list[str]) -> None:
    success: dict[str, list[int]] = load_checkpoint(SUCCESS_OUTPUT_FILE)
    failed:  dict[str, str]       = load_checkpoint(FAILED_OUTPUT_FILE)

    already_done = set(success.keys()) | set(failed.keys())
    pending = [s for s in smiles_list if s not in already_done]

    tqdm.write(
        f"Total: {len(smiles_list)} | "
        f"Already done: {len(already_done)} | "
        f"Remaining: {len(pending)} | "
        f"Rate limit: {REQUESTS_PER_MIN} req/min | "
        f"Workers: {MAX_WORKERS}"
    )

    if not pending:
        tqdm.write("Nothing to process. Exiting.")
        return

    processed_since_checkpoint = 0
    start_time = time.perf_counter()

    progress = tqdm(
        total=len(smiles_list),
        initial=len(already_done),
        desc="Mapping SMILES → CID",
        unit="drug",
        dynamic_ncols=True,
        colour="green",
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start : batch_start + BATCH_SIZE]

            future_to_smiles = {
                executor.submit(fetch_cids, smiles): smiles
                for smiles in batch
            }

            for future in as_completed(future_to_smiles):
                smiles, cids, error = future.result()

                if error is None:
                    success[smiles] = cids
                else:
                    failed[smiles] = error
                    log.warning(f"Failed: '{smiles[:40]}' — {error}")

                processed_since_checkpoint += 1

                elapsed = time.perf_counter() - start_time
                rate    = (len(success) + len(failed)) / elapsed if elapsed > 0 else 0
                progress.set_postfix(
                    success=len(success),
                    failed=len(failed),
                    rate=f"{rate:.1f}/s",
                    refresh=False,
                )
                progress.update(1)

                if processed_since_checkpoint >= CHECKPOINT_EVERY:
                    save_checkpoint(success, SUCCESS_OUTPUT_FILE)
                    save_checkpoint(failed,  FAILED_OUTPUT_FILE)
                    processed_since_checkpoint = 0

    progress.close()

    # Final save
    save_checkpoint(success, SUCCESS_OUTPUT_FILE)
    save_checkpoint(failed,  FAILED_OUTPUT_FILE)

    elapsed = time.perf_counter() - start_time
    tqdm.write(
        f"\nDone! Total: {len(smiles_list)} | "
        f"Success: {len(success)} | Failed: {len(failed)} | "
        f"Time: {elapsed:.1f}s"
    )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main() -> None:
    ood_datapath = '/DATA/DATANAS2/rhh25/dti_dataset/drugood_all/'
    ood_drug_df  = get_ood_data(ood_datapath)
    ood_drug_df  = ood_drug_df.dropna(subset=['smiles'])
    ood_drugs    = ood_drug_df.smiles.unique().tolist()

    kiba_datapath = '/DATA/DATANAS2/rhh25/dti_dataset/davis_kiba/kiba_all.csv'
    kiba_df       = pd.read_csv(kiba_datapath)
    kiba_df       = kiba_df.dropna(subset=['compound_iso_smiles'])
    kiba_drugs    = kiba_df.compound_iso_smiles.unique().tolist()

    smile_drugs = list(set(ood_drugs) | set(kiba_drugs))

    tqdm.write(f"Loaded {len(smile_drugs)} unique SMILES")
    process_all(smile_drugs)


if __name__ == "__main__":
    main()