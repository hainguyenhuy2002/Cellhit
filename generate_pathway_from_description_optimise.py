import json
import re
import concurrent.futures
from pathlib import Path

from CellHit.LLMs import generate_prompt
from CellHit.data import get_reactome_layers
from tqdm import tqdm
from vllm import LLM, SamplingParams

library_path = Path('/DATA/DATANAS2/rhh25/Cellhit/data') 

# ── LLM (vLLM — batched, fast) ────────────────────────────────────────────────
model_path = Path('/villa/rhh25/Cellhit/mixtral/').expanduser()

llm = LLM(
    model=str(model_path),
    revision="gptq-4bit-32g-actorder_True",
    quantization="gptq",
    dtype='float16',
    gpu_memory_utilization=0.95,
    enforce_eager=True,
)





# ── Paths ─────────────────────────────────────────────────────────────────────
data_path          = Path('/DATA/DATANAS2/rhh25/Cellhit/data/')

select_prompt_path = data_path / 'prompts' / 'mixtral_pathway_selector.txt'
drug_folder        = library_path/'drug_descriptions'
output_folder      = library_path / 'drug_pathways'
output_folder.mkdir(exist_ok=True)

# ── Tunable knobs ─────────────────────────────────────────────────────────────
BATCH_SIZE     = 16   # drugs per batched LLM call (tune to VRAM)
PATHWAY_NUMBER = 15   # pathways to select per drug
SELF_K         = 4    # self-consistency repetitions
TEMPERATURE    = 0.7
WRITE_WORKERS  = 8

# ── Reactome pathways ─────────────────────────────────────────────────────────
reactome_pathways = (
    get_reactome_layers(data_path / 'reactome', layer_number=1)['PathwayName']
    .str.strip()
    .tolist()
)
pathway_set = set(reactome_pathways)




# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(drug_description: str) -> str:
    return generate_prompt(
        select_prompt_path,
        drug_description=drug_description,
        pathways_list="\n".join(reactome_pathways),
        pathway_number=PATHWAY_NUMBER,
    )


# ── Output parser ─────────────────────────────────────────────────────────────
# Matches lines like:  Pathway 3: Signal Transduction
_PATHWAY_RE = re.compile(r'Pathway\s+\d+\s*:\s*(.+)', re.IGNORECASE)
_RATIONALE_RE = re.compile(r'Rationale\s+\d+\s*:\s*(.+)', re.IGNORECASE)


def parse_output(text: str) -> dict:
    """
    Extract (rationale, pathway) pairs from free-form LLM output.
    Falls back to fuzzy matching against the known pathway list if the
    exact name isn't found.
    """
    lines      = text.splitlines()
    pathways   = []
    rationales = []

    for line in lines:
        pm = _PATHWAY_RE.match(line.strip())
        if pm:
            name = pm.group(1).strip()
            # exact match first
            if name in pathway_set:
                pathways.append(name)
            else:
                # fuzzy: pick the pathway with the most overlapping words
                name_words = set(name.lower().split())
                best = max(reactome_pathways,
                           key=lambda p: len(set(p.lower().split()) & name_words))
                pathways.append(best)

        rm = _RATIONALE_RE.match(line.strip())
        if rm:
            rationales.append(rm.group(1).strip())

    # pad/trim to PATHWAY_NUMBER
    pathways   = (pathways   + [''] * PATHWAY_NUMBER)[:PATHWAY_NUMBER]
    rationales = (rationales + [''] * PATHWAY_NUMBER)[:PATHWAY_NUMBER]
    return {'pathway_names': pathways, 'pathway_rationales': rationales}


# ── Self-consistency aggregator ───────────────────────────────────────────────

def self_consistency(dict_list: list, normalize: bool = False) -> dict:
    out = {}
    for d in dict_list:
        for pathway, rationale in zip(d['pathway_names'], d['pathway_rationales']):
            if not pathway:
                continue
            if pathway not in out:
                out[pathway] = {'count': 0, 'rationales': []}
            out[pathway]['count'] += 1
            out[pathway]['rationales'].append(rationale)
    if normalize:
        n = len(dict_list)
        for v in out.values():
            v['count'] /= n
    return out


# ── File helpers ──────────────────────────────────────────────────────────────

def already_done(drug_name: str) -> bool:
    return (output_folder / f'{drug_name}_pathways.json').exists()


def _write_one(args):
    drug_name, result = args
    path = output_folder / f'{drug_name}_pathways.json'
    path.write_text(json.dumps(result, indent=2))


def write_batch(drug_names, results):
    with concurrent.futures.ThreadPoolExecutor(max_workers=WRITE_WORKERS) as pool:
        pool.map(_write_one, zip(drug_names, results))


# ── Core batch processor ──────────────────────────────────────────────────────

def process_batch(batch_drugs: list[tuple[str, str]]):
    """
    batch_drugs: list of (drug_name, drug_description)

    Strategy:
      - Run SELF_K rounds of batched inference, each round processing
        all drugs in the batch simultaneously.
      - Each round = 1 call to llm.generate() with len(batch) prompts.
      - Total LLM calls = SELF_K  (not SELF_K × len(batch)).
    """
    drug_names   = [d[0] for d in batch_drugs]
    prompts      = [build_prompt(d[1]) for d in batch_drugs]

    # dict_lists[i] accumulates self_k outputs for drug i
    dict_lists: list[list[dict]] = [[] for _ in range(len(batch_drugs))]

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=0.95,
        max_tokens=2048,   # pathway selection output is verbose
    )

    # SELF_K batched rounds — all drugs in parallel each round
    for _ in range(SELF_K):
        outputs = llm.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            text   = output.outputs[0].text
            parsed = parse_output(text)
            dict_lists[i].append(parsed)

    # Aggregate and write
    results = [self_consistency(dl) for dl in dict_lists]
    write_batch(drug_names, results)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Collect all pending drugs
    all_drugs = []
    for file in sorted(drug_folder.glob('*.txt')):
        drug_name = file.stem.replace('_description_refined', '')
        if not already_done(drug_name):
            all_drugs.append((drug_name, file.read_text()))

    print(f"{len(all_drugs)} drugs pending")

    # Split into mini-batches
    batches = [
        all_drugs[i:i + BATCH_SIZE]
        for i in range(0, len(all_drugs), BATCH_SIZE)
    ]

    for batch in tqdm(batches, desc='Batches', unit='batch'):
        process_batch(batch)

    print("All done.")