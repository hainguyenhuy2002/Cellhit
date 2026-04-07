import json
import gc
import torch
import os
import guidance
import pandas as pd
from guidance import select,gen,models
from pathlib import Path

#set current working directory to the library path
library_path = Path('/villa/rhh25/Cellhit/') #replace with yours
os.chdir(library_path)

from CellHit.LLMs import generate_prompt
from CellHit.data import get_reactome_layers

data_path = library_path/'data'
describe_prompt_path = data_path/'prompts'/'drug_description_prompt.txt'
refine_prompt_path = data_path/'prompts'/'drug_refiner_prompt.txt'
model_path = Path('~/Cellhit/mixtral/').expanduser() 

select_prompt_path = data_path/'prompts'/'mixtral_pathway_selector.txt'
model_path = Path('~/Cellhit/mixtral/').expanduser()


@guidance
def inference(lm,prompt, pathway_number, pathways_list,temperature=0.7):

    break_line = "\n"

    lm += prompt

    choosen_pathways = []
    #controlled generation
    for i in range(pathway_number):
        feasible_pathways = [i for i in pathways_list if i not in set(choosen_pathways)]
        lm += f"\nRationale {i+1}:{gen(f'rationale_{i+1}',stop=break_line,temperature=temperature)}"
        lm += f"\nPathway {i+1}:{select(options=feasible_pathways,name=f'pathway_{i+1}')}\n"
        choosen_pathways.append(lm[f'pathway_{i+1}'])
    return lm


def dictionary_maker(lm,k=15):

    out = {}
    out['pathway_names'] = []
    out['pathway_rationales'] = []

    for i in range(k):
        out['pathway_names'].append(lm[f'pathway_{i+1}'])
        out['pathway_rationales'].append(lm[f'rationale_{i+1}'])

    return out


def self_consistency(dict_list,normalize=False):

    out = {}

    for d in dict_list:
        
        for pathway,rationale in zip(d['pathway_names'],d['pathway_rationales']):
            if pathway not in out.keys():
                out[pathway] = {}
                out[pathway]['count'] = 1
                out[pathway]['rationales'] = [rationale]
            else:
                out[pathway]['count'] += 1
                out[pathway]['rationales'].append(rationale)

    #devide rationales by count
    if normalize:
        for key in out.keys():
            out[key]['count'] = out[key]['count']/len(dict_list)

    return out


reactome_pathways = get_reactome_layers(data_path/'reactome',layer_number=1)['PathwayName'].tolist()

gpu_id = 0
lm = models.Transformers(str(model_path),**{"device_map":f"cuda:{gpu_id}","revision":"gptq-4bit-32g-actorder_True"})

pathway_number = 15
self_k = 5
temperature = 0.7

# with open('drug_description_refined.txt','r') as f:
#     drug_description = f.read()

drug_folder = library_path/'drug_descriptions'


for file in drug_folder.glob('*.txt'):
    drug_description = file.read_text()
    drug_name = file.name.split('_')[0]

    prompt = generate_prompt(data_path/'prompts'/'mixtral_pathway_selector.txt',**{'drug_description':drug_description,'pathways_list':reactome_pathways,'pathway_number':pathway_number})

    dict_list = []

    for iter in range(self_k):
        lm = lm + inference(prompt,pathway_number=pathway_number, pathways_list=reactome_pathways,temperature=temperature)
        dict_list.append(dictionary_maker(lm))
        lm.reset()
        gc.collect()
        torch.cuda.empty_cache()

    output_dict = self_consistency(dict_list)


    with open(f"{data_path}/{drug_name}_pathways.json", "w") as f:
        json.dump(output_dict, f)