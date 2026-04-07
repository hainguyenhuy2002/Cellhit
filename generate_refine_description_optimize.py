import os
import concurrent.futures
import time
import threading

from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

from CellHit.LLMs import fetch_abstracts, generate_prompt
from CellHit.data import obtain_drugs_metadata
from utils.mapping import drugcomb_prism_mapping

# ── Paths ─────────────────────────────────────────────────────���───────────────
library_path         = Path('/villa/rhh25/Cellhit/')
data_path            = library_path / 'data'
describe_prompt_path = data_path / 'prompts' / 'drug_description_prompt.txt'
refine_prompt_path   = data_path / 'prompts' / 'drug_refiner_prompt.txt'
model_path           = Path('~/Cellhit/mixtral/').expanduser()

OUTPUT_DIR   = Path('data/drug_descriptions')
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Tunable knobs ─────────────────────────────────────────────────────────────
BATCH_SIZE       = 12   # number of drugs per mini-batch (tune to VRAM / RAM)
WRITE_WORKERS    = 4   # parallel threads for file writes
ABSTRACTS_K      = 10   # number of PubMed abstracts per drug
REQUESTS_PER_SEC = 3


# ── Dataset / metadata ───────────────────────────────────────────────���────────
dataset          = 'prism'
metadata         = obtain_drugs_metadata(dataset, data_path)
metadata = metadata.dropna(subset=['repurposing_target', 'MOA'])
df_drugcomb = pd.read_csv(data_path/'metadata'/'drugcombs_scored.csv')

drugcomb_drugs = set(df_drugcomb['Drug1']).union(set(df_drugcomb['Drug2']))
prism_drugs = set(metadata['Drug'])

drugcomb_to_prism_cleaned = drugcomb_prism_mapping(drugcomb_drugs, prism_drugs, data_path)
metadata = metadata[metadata['Drug'].isin(list(drugcomb_to_prism_cleaned))].reset_index(drop=True)
metadata_indexed = metadata.set_index('Drug')   # O(1) look-ups
drug_list        = metadata_indexed.index.tolist()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = LLM(
    model=str(model_path),
    revision="gptq-4bit-32g-actorder_True",
    quantization="gptq",
    dtype='float16',
    gpu_memory_utilization=1,
    enforce_eager=True,
    # **{"attn_implementation": "flash_attention_2"}  # Ampere+ GPUs only
)
sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=1024)


# ── Helpers ───────────────────────────────────────────────────────────────────

# def _fetch_one(drug_name, mail=None, k=10):
#     """Fetch PubMed abstracts for a single drug. Returns (name, list[str])."""
#     try:
#         abstracts = fetch_abstracts(drug_name, mail=mail, k=k)
#         return drug_name, [str(a) for a in abstracts]
#     except Exception as exc:
#         print(f"[WARN] fetch_abstracts failed for '{drug_name}': {exc}")
#         return drug_name, []


# def fetch_batch_abstracts(batch, mail=None, k=10):
#     """Fetch abstracts for every drug in *batch* concurrently."""
#     with concurrent.futures.ThreadPoolExecutor(max_workers=ABSTRACT_WORKERS) as pool:
#         results = pool.map(lambda name: _fetch_one(name, mail, k), batch)
#     return dict(results)


class RateLimiter:
    """
    Token-bucket rate limiter.
    Allows up to `rate` calls per second across all threads.
    """
    def __init__(self, rate: float):
        self.rate      = rate           # max calls/sec
        self.interval  = 1.0 / rate     # minimum gap between calls
        self._lock     = threading.Lock()
        self._last     = 0.0            # timestamp of last allowed call

    def acquire(self):
        """Block until it is safe to make the next request."""
        with self._lock:
            now  = time.monotonic()
            wait = self._interval_end - now
            if wait > 0:
                time.sleep(wait)
            self._interval_end = time.monotonic() + self.interval

    # initialise _interval_end lazily
    @property
    def _interval_end(self):
        return getattr(self, '__interval_end', 0.0)
    @_interval_end.setter
    def _interval_end(self, v):
        self.__interval_end = v


# One shared limiter for the whole process
_rate_limiter = RateLimiter(rate=REQUESTS_PER_SEC)


def _fetch_one(drug_name, mail=None, k=10):
    """
    Fetch PubMed abstracts for a single drug, honouring the NCBI rate limit.
    Returns (drug_name, list[str]).
    """
    _rate_limiter.acquire()          # ← blocks until our slot is available
    try:
        abstracts = fetch_abstracts(drug_name, mail=mail, k=k)
        return drug_name, [str(a) for a in abstracts]
    except Exception as exc:
        print(f"[WARN] fetch_abstracts failed for '{drug_name}': {exc}")
        return drug_name, []


def fetch_batch_abstracts(batch, mail=None, k=10):
    """
    Fetch abstracts for all drugs in *batch* concurrently,
    but no faster than NCBI allows.

    Worker count is capped at REQUESTS_PER_SEC so we never have more
    in-flight requests than the API permits per second.
    """
    max_workers = REQUESTS_PER_SEC  # e.g. 3 or 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, name, mail, k): name for name in batch}
        results = {}
        for fut in concurrent.futures.as_completed(futures):
            name, abstracts = fut.result()
            results[name]   = abstracts
    return results




def build_describe_prompt(drug_name, metadata_indexed, describe_prompt_path):
    row    = metadata_indexed.loc[drug_name]
    moa    = row['MOA']                if isinstance(row['MOA'], str)                else "Not annotated"
    target = row['repurposing_target'] if isinstance(row['repurposing_target'], str) else "Not annotated"
    return generate_prompt(describe_prompt_path,
                           drug_name=drug_name,
                           mechanism_of_action=moa,
                           putative_targets=target)


def build_refine_prompt(description, abstracts, refine_prompt_path):
    formatted = "\n".join(f"Abstract {i+1}: {a}" for i, a in enumerate(abstracts))
    return generate_prompt(refine_prompt_path,
                           previous_output=description,
                           formatted_abstracts=formatted)


def _write_one(drug_name, text):
    (OUTPUT_DIR / f'{drug_name}_description_refined.txt').write_text(text)


def write_batch(drug_names, refined_texts):
    """Write output files for a batch concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=WRITE_WORKERS) as pool:
        pool.map(_write_one, drug_names, refined_texts)


def already_done(drug_name):
    """Skip drugs whose output file already exists (safe resumption)."""
    return (OUTPUT_DIR / f'{drug_name}_description_refined.txt').exists()


# ── Batched pipeline ──────────────────────────────────────────────────────────

def process_batch(batch):
    """
    For a list of drug names:
      1. Fetch abstracts in parallel (I/O).
      2. One batched LLM call  →  descriptions.
      3. One batched LLM call  →  refined descriptions.
      4. Write output files in parallel (I/O).
    """


    # --- 1. Parallel abstract fetch ------------------------------------------
    abstracts_map = fetch_batch_abstracts(batch, mail=None, k=ABSTRACTS_K)

    # --- 2. Build describe prompts & single batched inference ----------------
    describe_prompts = [
        build_describe_prompt(name, metadata_indexed, describe_prompt_path)
        for name in batch
    ]
    describe_outputs = llm.generate(describe_prompts, sampling_params)
    descriptions     = [o.outputs[0].text for o in describe_outputs]

    # --- 3. Build refine prompts & single batched inference ------------------
    refine_prompts = [
        build_refine_prompt(desc, abstracts_map.get(name, []), refine_prompt_path)
        for name, desc in zip(batch, descriptions)
    ]
    refine_outputs = llm.generate(refine_prompts, sampling_params)
    refined_texts  = [o.outputs[0].text for o in refine_outputs]

    # --- 4. Parallel file writes ---------------------------------------------
    write_batch(batch, refined_texts)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Filter out already-completed drugs so the run is safely resumable
    pending = [d for d in drug_list if not already_done(d)]
    print(f"{len(drug_list)} drugs total | {len(pending)} pending | "
          f"{len(drug_list) - len(pending)} already done")

    # Split into mini-batches
    batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Batches", unit="batch"):
        process_batch(batch)

    print("All done.")