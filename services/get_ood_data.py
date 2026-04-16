import json
import os
import glob
import pandas as pd

from pathlib import Path


def get_ood_data(odd_directory):

    def get_dataframe(data):
        rows = []
        for key in ['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']:
            for entry in data[key]:
                rows.append({
                    'smiles': entry['smiles'],
                    'reg_label': entry['reg_label'],
                    'assay_id': entry['assay_id'],
                    'protein': entry['protein'],
                    'cls_label': entry['cls_label'],
                    'domain_id': entry['domain_id']
                })

        # Create DataFrame
        df = pd.DataFrame(rows)
        return df


    #directory = '/DATA/DATANAS2/rhh25/dti_dataset/drugood_all/'

    all_json_files = glob.glob(os.path.join(odd_directory, '*.json'))
    keywords = ['core', 'ec50', 'sbap']
    matching_files = [f for f in all_json_files if all(k in f for k in keywords)]

    with open(matching_files[0], "r") as f:
        core_ec50_assay = json.load(f)
    with open(matching_files[1], "r") as f:
        core_ec50_protein = json.load(f)
    with open(matching_files[2], "r") as f:
        core_ec50_protein_fam = json.load(f)
    with open(matching_files[3], "r") as f:
        core_ec50_scaffold = json.load(f)
    with open(matching_files[4], "r") as f:
        core_ec50_size = json.load(f)

    assay_df = get_dataframe(core_ec50_assay['split'])
    protein_df = get_dataframe(core_ec50_protein['split'])
    protein_fam_df = get_dataframe(core_ec50_protein_fam['split'])
    scaffold_df = get_dataframe(core_ec50_scaffold['split'])
    size_df = get_dataframe(core_ec50_size['split'])

    over_all_df = pd.concat([assay_df, protein_df, protein_fam_df, scaffold_df, size_df], ignore_index=True)

    #unique_drug = over_all_df.smiles.unique()
    return over_all_df