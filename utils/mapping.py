import json

def drugcomb_prism_mapping(drugcomb_drugs, prism_drugs, data_path):
    drugcomb_to_prism = {}
    for drugcomb_drug in drugcomb_drugs:
        drug_name = str(drugcomb_drug).strip()
        drug_name_upper = drug_name.upper()

        if drug_name_upper in prism_drugs:
            drugcomb_to_prism[drugcomb_drug] = drug_name_upper
            continue


        for prism_drug in prism_drugs:
            prism_drug_upper = str(prism_drug).strip().upper()
            if '-' not in drug_name:
                if (drug_name_upper in prism_drug_upper) or (prism_drug_upper in drug_name_upper):
                    drugcomb_to_prism[drugcomb_drug] = prism_drug
                    break
    
    with open(data_path/'metadata'/'drugcomb_to_prism_mapping.json', 'w') as f:
        json.dump(drugcomb_to_prism, f, indent=4)


    return drugcomb_to_prism.values()