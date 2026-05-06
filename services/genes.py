import re
import pandas as pd


def map_cellline_to_modelid(cellline_name, mapping_dict):
    """Map DrugComb cell line name to DepMap ModelID"""
    cellline_upper = cellline_name.upper()
    
    # Exact match
    if cellline_upper in mapping_dict:
        return mapping_dict[cellline_upper]
    
    # Partial match (if exact not found)
    for mapped_name, model_id in mapping_dict.items():
        if cellline_upper in mapped_name or mapped_name in cellline_upper:
            return model_id
    
    return None


def get_active_genes(df_drugcombs, metadata, ccle_expression, threshold=1000):
    """Identify active genes based on mean expression across cell lines"""
    # Create mapping: StrippedCellLineName -> ModelID
    cell_line_list = df_drugcombs['Cell line'].unique()
    print("Number of unique cell lines in DrugComb:", len(cell_line_list))

    cellline_to_modelid = dict(zip(
        metadata['StrippedCellLineName'].str.upper(),
        metadata['ModelID']
    ))


    df_drugcombs['CellLineClean'] = df_drugcombs['Cell line'].apply(lambda x: re.sub(r'[-\s\/\\]', '', x.strip().upper()))
    df_drugcombs['CellLineClean'] = df_drugcombs['CellLineClean'].apply(lambda x: 'HL60' if x in ['HL60(TB)'] else x)

    df_drugcombs['ModelID'] = df_drugcombs['CellLineClean'].apply(
        lambda x: map_cellline_to_modelid(x, cellline_to_modelid)
    )

    # Extract metadata columns
    metadata_cols = ['SequencingID', 'ModelID', 'IsDefaultEntryForModel', 
                    'ModelConditionID', 'IsDefaultEntryForMC']

    # Extract gene expression columns (everything else)
    expression_cols = [col for col in ccle_expression.columns if col not in metadata_cols]

    print(f"Gene expression columns: {len(expression_cols)}")
    ccle_expression_filter = ccle_expression[ccle_expression['ModelID'].isin(df_drugcombs['ModelID'].dropna().unique())].set_index('ModelID')[expression_cols]

    active_genes =  ccle_expression_filter.columns[ccle_expression_filter.mean(numeric_only=True) > threshold].tolist()
    print(f"Number of active genes: {len(active_genes)}")


    number_celllines = len(ccle_expression_filter.T.columns[ccle_expression_filter[active_genes].T.mean(numeric_only=True)>0])
    print("Number of cell lines with active genes: ", number_celllines)


    ## remove parentheses in and content in columns name
    ccle_expression_filter = ccle_expression_filter[active_genes]
    ccle_expression_filter.columns = ccle_expression_filter.columns.str.replace(r'\(.*\)', '', regex=True).str.strip()

    active_genes = ccle_expression_filter.columns.tolist()
    return active_genes, ccle_expression_filter, df_drugcombs


def map_uniprot_to_gene(mg, uniprot_ids, batch_size=1000):
    all_results = []

    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i+batch_size]
        print(f"Processing batch {i} → {i+len(batch)}")

        res = mg.querymany(
            batch,
            scopes="uniprot",
            fields="symbol",
            species="human"
        )

        all_results.extend(res)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Keep relevant columns
    df = df[["query", "symbol"]]

    # Rename columns
    df.columns = ["UniProt", "Gene"]

    return df