import time
import pandas as pd
from pathlib import Path
from mygene import MyGeneInfo
from tqdm import tqdm
from services.drug import get_drugcomb_smiles

data_path = Path('/DATA/DATANAS2/rhh25/dti_dataset/')


drugcomb_smile_df = get_drugcomb_smiles()



targetDeca_df = pd.read_csv(data_path/'biosnap/ChG-TargetDecagon_targets.csv')
targetDeca_df['# Drug\tGene'] = targetDeca_df['# Drug\tGene'].apply(lambda x: str(x))
targetDeca_df = targetDeca_df['# Drug\tGene'].str.split('\t', expand=True).reset_index()
targetDeca_df.columns = ['Drug', 'Gene']


## With DrugBank drugs
miner_chem_df = pd.read_csv(data_path/'biosnap/ChG-Miner_miner-chem-gene.tsv')
miner_chem_df[['Drug', 'Gene']] = miner_chem_df['#Drug\tGene'].str.split('\t', expand=True)
miner_chem_df.drop(columns=['#Drug\tGene'], inplace=True)
miner_chem_df.columns = ['DrugBankId', 'UniprotGene']
drugbank_lst = miner_chem_df['DrugBankId'].unique().tolist()


syn_df = pd.read_csv(data_path/'biosnap/CID-Synonym-filtered', sep="\t", header=None)
syn_df.columns = ['CID', 'DrugBankId']
syn_df_filter = syn_df[syn_df['DrugBankId'].isin(drugbank_lst)]


biosnap_df = pd.merge(syn_df_filter, drugcomb_smile_df, on='CID', how='inner')
biosnap_df = pd.merge(biosnap_df, miner_chem_df, on='DrugBankId', how='inner')
unique_uniprot_ids = biosnap_df['UniprotGene'].unique().tolist()


mg = MyGeneInfo()

def map_uniprot_to_gene(uniprot_ids, batch_size=50, sleep_time=1):
    all_results = []

    # tqdm progress bar over batches
    for i in tqdm(range(0, len(uniprot_ids), batch_size)):
        batch = uniprot_ids[i:i+batch_size]

        res = mg.querymany(
            batch,
            scopes="uniprot",
            fields="symbol",
            species="human"
        )

        all_results.extend(res)

        # avoid rate limit
        time.sleep(sleep_time)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Keep useful columns
    df = df[["query", "symbol"]]

    # Rename
    df.columns = ["UniProt", "Gene"]

    return df


# Usage

# Example usage
uniprot_ids = unique_uniprot_ids
mapping_df = map_uniprot_to_gene(uniprot_ids)

mapping_df.to_csv(data_path/'biosnap/uniprot_to_gene_symbol_mapping.csv', index=False)