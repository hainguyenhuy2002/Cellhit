import mygene
import json

def map_ensembl_proteins_to_genes(ensembl_protein_ids):
    """
    Convert Ensembl protein IDs (ENSP...) to gene symbols
    
    Input format: ['9606.ENSP00000002233', '9606.ENSP00000000412', ...]
    Output: {'ENSP00000002233': 'TSPAN6', 'ENSP00000000412': 'DPM1', ...}
    """
    print(f"\nMapping {len(ensembl_protein_ids)} Ensembl proteins to genes...")
    
    ensp_ids = [p.split('.')[-1] for p in ensembl_protein_ids]
    
    mg = mygene.MyGeneInfo()
    
    protein_to_gene = {}
    
    # Process in batches
    batch_size = 300
    for i in range(0, len(ensp_ids), batch_size):
        batch = ensp_ids[i:i+batch_size]
        
        try:
            # Query by Ensembl protein ID
            results = mg.querymany(
                batch,
                scopes='ensemblprotein',  # Query by Ensembl protein ID
                species='human',
                fields='symbol',  # Get gene symbol
                as_dataframe=False,
                returnall=False
            )
            
            for result in results:
                if 'symbol' in result:
                    ensp_id = f'9606.{result["query"]}'
                    gene_symbol = result['symbol']
                    protein_to_gene[ensp_id] = gene_symbol
        
        except Exception as e:
            #print(f"Error in batch {i//batch_size}: {e}")
            continue
    
    with open("data/metadata/protein_to_gene_mapping.json", "w") as f:
        json.dump(protein_to_gene, f, indent=4)

    print(f"Successfully mapped {len(protein_to_gene)} proteins to genes")
    
    return protein_to_gene