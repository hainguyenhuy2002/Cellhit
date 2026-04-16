import json

from utils.dicts import merge_dicts
import pandas as pd


def get_drugcomb_smiles():

    with open("data/drug_smiles.json", "r") as f:
        smiles_drugs = json.load(f)

    with open("data/drug_smiles1.json", "r") as f:
        smiles_drugs1 = json.load(f)

    with open("data/drug_smiles2.json", "r") as f:
        smiles_drugs2 = json.load(f)

    final_smile_drugs = merge_dicts(smiles_drugs, smiles_drugs1, smiles_drugs2)

    drugcomb_smile_df = pd.DataFrame.from_dict(final_smile_drugs, orient='index').reset_index()

    drugcomb_smile_df.columns = ['Drug Name', 'CID', 'Smiles']

    return drugcomb_smile_df