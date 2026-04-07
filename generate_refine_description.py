import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path

from vllm import LLM, SamplingParams
from CellHit.LLMs import fetch_abstracts, generate_prompt
from CellHit.data import obtain_drugs_metadata

library_path = Path('/villa/rhh25/Cellhit/') 
data_path = library_path/'data'
describe_prompt_path = data_path/'prompts'/'drug_description_prompt.txt'
refine_prompt_path = data_path/'prompts'/'drug_refiner_prompt.txt'
model_path = Path('~/Cellhit/mixtral/').expanduser()

#Read dataset and metadata
dataset = 'gdsc'
metadata = obtain_drugs_metadata(dataset,data_path)
drug_list = metadata['Drug'].tolist()

#Initialize the LLM
llm = LLM(
            model=str(model_path),
            revision="gptq-4bit-32g-actorder_True",
            quantization="gptq",
            dtype='float16',
            gpu_memory_utilization=1,
            enforce_eager=True)
            #**{"attn_implementation":"flash_attention_2"}) Turn on for faster decoding (Ampere and newer GPUs only)

#initialize sampling parameters
sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=1024)


def get_refined_drug_description(drug_name, metadata, llm, describe_prompt_path, refine_prompt_path, sampling_params):
    mechanism_of_action = metadata.loc[metadata['Drug']==drug_name]['MOA'].values[0]
    putative_target = metadata.loc[metadata['Drug']==drug_name]['repurposing_target'].values[0]

    #fetch abstracts
    abstracts = fetch_abstracts(drug_name,mail=None,k=10)
    abstracts = [str(a) for a in abstracts]


    describe_prompt = generate_prompt(describe_prompt_path,**{'drug_name':drug_name, 'mechanism_of_action':mechanism_of_action, 'putative_targets':putative_target})

    outputs = llm.generate(describe_prompt, sampling_params)
    drug_description = outputs[0].outputs[0].text

    refine_prompt = generate_prompt(refine_prompt_path,**{'previous_output':drug_description,'formatted_abstracts':"\n".join([f"Abstract {i+1}: {abstract}" for i, abstract in enumerate(abstracts)])})

    outputs = llm.generate(refine_prompt, sampling_params)
    refined_drug_description = outputs[0].outputs[0].text
    #dump refined_drug_description to a file
    with open(f'{drug_name}_description_refined.txt','w') as f:
        f.write(refined_drug_description)


#generate refined drug descriptions for all drugs in the dataset
for drug_name in drug_list:
    get_refined_drug_description(drug_name, metadata, llm, describe_prompt_path, refine_prompt_path, sampling_params)   