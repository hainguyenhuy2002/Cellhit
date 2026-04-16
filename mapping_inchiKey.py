import asyncio
import time
import json
import logging
import pandas as pd
import aiohttp
from pathlib import Path
from tqdm.asyncio import tqdm



data_path = Path('/DATA/DATANAS2/rhh25/Cellhit/data/')

df_drugcombs = pd.read_csv(f'{data_path}/metadata/drugcombs_scored.csv')
df_drugcombs = df_drugcombs.dropna(subset=['Drug1', 'Drug2'] )
unique_drugs =  set(df_drugcombs.Drug1.unique().tolist()).union(
        set(df_drugcombs.Drug2.unique().tolist())
    )



# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PUBCHEM_BATCH_SIZE  = 25       # Max names per PubChem POST request
CHEMBL_CONCURRENCY  = 2        # Max parallel ChEMBL requests
PUBCHEM_RPS         = 3         # Max requests/sec (PubChem limit)
MAX_RETRIES         = 5         # Retries on 429 / network error
BACKOFF_BASE        = 2.0       # Exponential backoff base (seconds)
CHECKPOINT_SIZE     = 100        # Save to JSON every N drugs processed

OUTPUT_MAPPED_FILE  = "drug_mapped.json"
OUTPUT_FAILED_FILE  = "drug_failed.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. CHECKPOINT MANAGER
# ─────────────────────────────────────────────
class CheckpointManager:
    """
    Manages two JSON checkpoint files:
      - mapped_file : { drug_name -> inchikey }
      - failed_file : [ drug_name, ... ]

    Every time `add()` is called, the in-memory state is updated.
    Every CHECKPOINT_SIZE additions, both files are flushed to disk.
    Call `flush()` at the end to ensure the final state is persisted.
    """

    def __init__(
        self,
        mapped_file: str,
        failed_file: str,
        checkpoint_size: int = CHECKPOINT_SIZE,
    ):
        self.mapped_file     = Path(mapped_file)
        self.failed_file     = Path(failed_file)
        self.checkpoint_size = checkpoint_size

        # Load existing checkpoints so interrupted runs resume cleanly
        self.mapped: dict[str, str]  = self._load_json(self.mapped_file, default={})
        self.failed: list[str]       = self._load_json(self.failed_file, default=[])
        self._counter                = 0

        if self.mapped or self.failed:
            log.info(
                "Resumed from checkpoint — mapped: %d, failed: %d",
                len(self.mapped), len(self.failed),
            )

    # ── public API ───────────────────────────────────────────────────────
    def already_processed(self, name: str) -> bool:
        return name in self.mapped or name in self.failed

    def add_mapped(self, name: str, inchikey: str):
        self.mapped[name] = inchikey
        self._tick()

    def add_failed(self, name: str):
        if name not in self.failed:
            self.failed.append(name)
        self._tick()

    def flush(self):
        """Force-write both files regardless of counter."""
        self._write(self.mapped_file, self.mapped)
        self._write(self.failed_file, self.failed)
        log.info(
            "Checkpoint saved — mapped: %d, failed: %d",
            len(self.mapped), len(self.failed),
        )

    # ── internals ────────────────────────────────────────────────────────
    def _tick(self):
        self._counter += 1
        if self._counter >= self.checkpoint_size:
            self.flush()
            self._counter = 0

    @staticmethod
    def _load_json(path: Path, default):
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                log.warning("Could not read %s — starting fresh.", path)
        return default

    @staticmethod
    def _write(path: Path, data):
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)          # atomic rename — no half-written files


# ─────────────────────────────────────────────
# 2. RATE LIMITER
# ─────────────────────────────────────────────
class RateLimiter:
    """Token-bucket rate limiter for async code."""
    def __init__(self, rps: float):
        self._interval = 1.0 / rps
        self._last     = 0.0
        self._lock     = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now  = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


# ─────────────────────────────────────────────
# 3. RETRY HELPER
# ─────────────────────────────────────────────
async def fetch_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    limiter: RateLimiter,
    **kwargs,
) -> dict | None:
    """GET/POST with exponential backoff on 429 / 5xx errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        await limiter.acquire()
        try:
            async with getattr(session, method)(url, **kwargs) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                if resp.status == 404:
                    return None
                if resp.status == 429 or resp.status >= 500:
                    wait = BACKOFF_BASE ** attempt
                    log.warning(
                        "HTTP %s — attempt %d/%d, retry in %.1fs",
                        resp.status, attempt, MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            wait = BACKOFF_BASE ** attempt
            log.warning("Network error (%s) — retry in %.1fs", exc, wait)
            await asyncio.sleep(wait)
    log.error("All retries exhausted for %s", url)
    return None


# ─────────────────────────────────────────────
# 4. PUBCHEM LOOKUPS
# ─────────────────────────────────────────────
async def pubchem_single(
    session: aiohttp.ClientSession,
    name: str,
    limiter: RateLimiter,
) -> str | None:
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{aiohttp.helpers.quote(name, safe='')}/property/InChIKey/JSON"
    )
    payload = await fetch_with_retry(session, "get", url, limiter)
    if payload:
        props = payload.get("PropertyTable", {}).get("Properties", [])
        if props:
            return props[0].get("InChIKey")
    return None


async def pubchem_batch_chunk(
    session: aiohttp.ClientSession,
    names: list[str],
    limiter: RateLimiter,
) -> dict[str, str]:
    """
    Batch-POST up to PUBCHEM_BATCH_SIZE names.
    Returns {name: inchikey} for every name that was resolved.
    Falls back to individual GET for any name the batch missed.
    """
    result: dict[str, str] = {}

    url  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/property/InChIKey/JSON"
    data = aiohttp.FormData()
    data.add_field("name", ",".join(names))

    payload = await fetch_with_retry(session, "post", url, limiter, data=data)
    cid_to_key: dict[str, str] = {}

    if payload:
        for prop in payload.get("PropertyTable", {}).get("Properties", []):
            key = prop.get("InChIKey")
            cid = str(prop.get("CID", ""))
            if key and cid:
                cid_to_key[cid] = key

    # Reverse-map each name → CID → InChIKey
    for name in names:
        cid_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{aiohttp.helpers.quote(name, safe='')}/cids/TXT"
        )
        await limiter.acquire()
        try:
            async with session.get(cid_url) as r:
                if r.status == 200:
                    cid = (await r.text()).strip()
                    if cid in cid_to_key:
                        result[name] = cid_to_key[cid]
        except Exception:
            pass

    # Individual GET fallback for anything still missing
    missing = [n for n in names if n not in result]
    for name in missing:
        key = await pubchem_single(session, name, limiter)
        if key:
            result[name] = key

    return result


# ─────────────────────────────────────────────
# 5. CHEMBL LOOKUPS
# ─────────────────────────────────────────────
async def chembl_single(
    session: aiohttp.ClientSession,
    name: str,
    limiter: RateLimiter,
) -> str | None:
    url = (
        f"https://www.ebi.ac.uk/chembl/api/data/molecule"
        f"?pref_name__iexact={aiohttp.helpers.quote(name, safe='')}&format=json"
    )
    payload = await fetch_with_retry(session, "get", url, limiter)
    if payload:
        molecules = payload.get("molecules", [])
        if molecules:
            structs = molecules[0].get("molecule_structures") or {}
            return structs.get("standard_inchi_key")
    return None


async def chembl_batch(
    session: aiohttp.ClientSession,
    names: list[str],
    limiter: RateLimiter,
) -> dict[str, str]:
    sem     = asyncio.Semaphore(CHEMBL_CONCURRENCY)
    results = {}

    async def lookup_one(name: str):
        async with sem:
            key = await chembl_single(session, name, limiter)
            if key:
                results[name] = key

    await asyncio.gather(*[lookup_one(n) for n in names])
    return results


# ─────────────────────────────────────────────
# 6. MAIN ASYNC PIPELINE
# ─────────────────────────────────────────────
async def map_all_drugs(
    drug_names: list[str],
    checkpoint: CheckpointManager,
):
    # Skip drugs already handled in a previous run
    pending = [n for n in drug_names if not checkpoint.already_processed(n)]
    log.info(
        "Total: %d | Already processed: %d | Pending: %d",
        len(drug_names), len(drug_names) - len(pending), len(pending),
    )

    if not pending:
        log.info("Nothing to do — all drugs already processed.")
        return

    pubchem_limiter = RateLimiter(PUBCHEM_RPS)
    chembl_limiter  = RateLimiter(CHEMBL_CONCURRENCY)

    timeout   = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=20)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        # ── Step 1: PubChem batch ──────────────────────────────────────────
        log.info("Step 1/2 — PubChem batch lookup (batch_size=%d)", PUBCHEM_BATCH_SIZE)
        pubchem_unresolved: list[str] = []

        chunks = [
            pending[i: i + PUBCHEM_BATCH_SIZE]
            for i in range(0, len(pending), PUBCHEM_BATCH_SIZE)
        ]

        for chunk in tqdm(chunks, desc="PubChem batches"):
            found = await pubchem_batch_chunk(session, chunk, pubchem_limiter)
            for name in chunk:
                if name in found:
                    checkpoint.add_mapped(name, found[name])
                else:
                    pubchem_unresolved.append(name)

        # ── Step 2: ChEMBL fallback ────────────────────────────────────────
        log.info(
            "Step 2/2 — ChEMBL fallback for %d unresolved drugs",
            len(pubchem_unresolved),
        )

        if pubchem_unresolved:
            # Process in CHECKPOINT_SIZE chunks so we checkpoint regularly
            chembl_chunks = [
                pubchem_unresolved[i: i + CHECKPOINT_SIZE]
                for i in range(0, len(pubchem_unresolved), CHECKPOINT_SIZE)
            ]
            for chunk in tqdm(chembl_chunks, desc="ChEMBL fallback"):
                found = await chembl_batch(session, chunk, chembl_limiter)
                for name in chunk:
                    if name in found:
                        checkpoint.add_mapped(name, found[name])
                    else:
                        checkpoint.add_failed(name)

    # Final flush — persist whatever remains below the checkpoint threshold
    checkpoint.flush()


# ─────────────────────────────────────────────
# 7. LOAD & SUMMARY HELPERS
# ─────────────────────────────────────────────
def load_drug_names(filepath: str) -> list[str]:
    df        = pd.read_csv(filepath)
    drug_cols = [c for c in ["drug_row", "drug_col"] if c in df.columns]
    if not drug_cols:
        raise ValueError(f"No drug columns found. Got: {df.columns.tolist()}")
    names = pd.unique(df[drug_cols].values.ravel("K"))
    return sorted({str(n).strip() for n in names if pd.notna(n) and str(n).strip()})


def print_summary(checkpoint: CheckpointManager, elapsed: float):
    mapped  = len(checkpoint.mapped)
    failed  = len(checkpoint.failed)
    total   = mapped + failed
    print(f"\n{'─'*42}")
    print(f"Total processed : {total}")
    print(f"Mapped          : {mapped} ({mapped / total * 100:.1f}%)")
    print(f"Failed          : {failed} ({failed / total * 100:.1f}%)")
    print(f"Elapsed         : {elapsed:.1f}s")
    print(f"{'─'*42}")
    print(f"Mapped  → {checkpoint.mapped_file}")
    print(f"Failed  → {checkpoint.failed_file}")
    if checkpoint.failed:
        print(f"\nFailed drugs ({failed}):")
        for d in checkpoint.failed:
            print(f"  - {d}")


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_FILE = "drugcomb_data.csv"

    checkpoint = CheckpointManager(
        mapped_file     = OUTPUT_MAPPED_FILE,
        failed_file     = OUTPUT_FAILED_FILE,
        checkpoint_size = CHECKPOINT_SIZE,
    )

    t0         = time.time()
    #drug_names = load_drug_names(INPUT_FILE)
    drug_names = list(unique_drugs)
    log.info("Loaded %d unique drug names", len(drug_names))

    asyncio.run(map_all_drugs(drug_names, checkpoint))
    print_summary(checkpoint, time.time() - t0)