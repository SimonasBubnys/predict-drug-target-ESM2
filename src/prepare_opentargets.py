import csv
import glob
import json
import os

import pandas as pd
import requests
from tqdm import tqdm

from src.embeddings import compute_drug_embedding, compute_target_embedding
from src.utils import COLLECTIONS, log, get_smiles_for_drug, get_seq_for_target
from src.vectordb import init_vectordb

# NOTE: Download opentargets before running this script
# ./scripts/download_opentargets.sh

# A list of KNOWN drugs-interacts_with-targets (from opentarget)
# Once we have this list, we just need to pass it to the compute_drug_embedding or compute_target_embedding functions
# These functions returns a dataframe with a "drug" column for the ID, and all other columns are the embeddings
# knownInteraction [ drug_id - target_id - 0 or 1 if interacts or not] (or could even be the mechanism of action string)
# target_df[id - embeddings]
# drugs_df[id - embeddings]
# TODO: First we get the df of knownInteraction, then generate list of drugs, pass it to function to calculate embed, same for targets


# Output file path
output_file_path = "../data/opentargets/merged_parsed.csv"


def get_jsonl_files(target_directory):
    """Return a list of JSONL files from the target directory."""
    return glob.glob(os.path.join(target_directory, "*.json"))


def extract_data_from_jsonl(filename):
    """Extract drugId and targetId from a JSONL file."""
    with open(filename) as file:
        for line in file:
            data = json.loads(line.strip())
            yield data.get("drugId", None), data.get("targetId", None)


def write_to_csv(output_csv, data, header=None):
    """Write the provided data to a CSV file."""
    with open(output_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if header:
            csv_writer.writerow(header)
        if isinstance(data, list):
            for row in data:
                csv_writer.writerow(row)
        elif isinstance(data, dict):
            for row in data.values():
                csv_writer.writerow(row.values())


def prepare(target_directory, output_directory):
    """Main function to orchestrate the extraction and saving process."""
    known_drug_targets = []

    # first extract the drug-target pairs from the opentargets json files
    json_files = get_jsonl_files(target_directory)
    for json_file in tqdm(json_files, desc="Processing files"):
        # log.info(json_file)
        for drugId, targetId in extract_data_from_jsonl(json_file):
            known_drug_targets.append(
                {
                    "drug": f"CHEMBL.COMPOUND:{drugId}",
                    "target": f"ENSEMBL:{targetId}",
                }
            )

    vectordb = init_vectordb(COLLECTIONS, recreate=False)

    df_known_dt = pd.DataFrame(known_drug_targets)

    # Get SMILES for drugs
    list_targets_no_seq = []
    targets_seq_list = []
    for target_id in tqdm(set(df_known_dt["drug"].tolist()), desc="Get drugs SMILES"):
        try:
            target_seq, target_label = get_smiles_for_drug(target_id)
            targets_seq_list.append({
                "drug": target_id,
                "smiles": target_seq,
                "label": target_label,
            })
        except Exception as e:
            list_targets_no_seq.append(target_id)
            # log.info(f"No SMILES for {target_id}")
    if list_targets_no_seq:
        log.info(list_targets_no_seq)
        log.info(f"⚠️ We could not find SMILES for {len(list_targets_no_seq)} drugs")
    df_drugs_smiles = pd.DataFrame(targets_seq_list)

    # Get AA seq for targets
    list_targets_no_seq = []
    targets_seq_list = []
    for target_id in tqdm(set(df_known_dt["target"].tolist()), desc="Get targets AA sequences"):
        try:
            target_seq, target_label = get_seq_for_target(target_id)
            targets_seq_list.append({
                "target": target_id,
                "sequence": target_seq,
                "label": target_label,
            })
        except Exception as e:
            list_targets_no_seq.append(target_id)
            # log.info(f"No AA seq for {target_id}")
    if list_targets_no_seq:
        log.info(list_targets_no_seq)
        log.info(f"⚠️ We could not find AA sequence for {len(list_targets_no_seq)} targets")
    df_targets_seq = pd.DataFrame(targets_seq_list)

    # Save DFs with SMILES to CSV files
    os.makedirs("data/opentargets", exist_ok=True)
    df_known_dt.to_csv('data/opentargets/known_drug_targets.csv', index=False)
    df_drugs_smiles.to_csv('data/opentargets/drugs_smiles.csv', index=False)
    df_targets_seq.to_csv('data/opentargets/targets_sequences.csv', index=False)


    # These functions retrieves SMILES and compute embeddings in 1 batch
    # df_drugs = compute_drug_embedding(vectordb, set(df_known_dt["drug"].tolist()))
    # log.info("DRUGS EMBEDDINGS COMPUTED")
    # df_targets = compute_target_embedding(vectordb, set(df_known_dt["target"].tolist()))
    # log.info("TARGETS EMBEDDINGS COMPUTED")

    # df_drugs.to_csv('data/opentargets/drugs_embeddings.csv', index=False)
    # df_targets.to_csv('data/opentargets/targets_embeddings.csv', index=False)



    # write_to_csv(f'{output_directory}/opentargets_drug_targets.csv',
    #                 drug_targets, ['DrugId', 'TargetId'])

    # # create a unique list of drug ids and target ids
    # drug_ids = list(set([t[0] for t in drug_targets]))
    # target_ids = list(set([t[1] for t in drug_targets]))

    # # now retrieve smiles and sequences
    # hashes = {}
    # file = f'{output_directory}/opentargets_drugs.csv'
    # # if os.path.exists(file):
    # with open(file) as csvfile:
    #     df = pd.read_csv(csvfile)
    #     hashes = df.to_dict('index')
    # print(df)
    # invalid_smiles = []
    # for drugId in tqdm(drug_ids, desc="Processing drugs"):
    #     if drugId in df['drug_id'].values \
    #         or drugId in invalid_smiles:
    #         continue

    #     smiles = get_smiles_from_drugId(drugId)
    #     if smiles is None:
    #         invalid_smiles.append(drugId)
    #         continue

    #     h = hash(smiles)
    #     if h in hashes:
    #         # retrieve the object
    #         o = hashes[h]
    #         if drugId not in o['other_ids']:
    #             o['other_ids'].append(drugId)
    #         hashes[h] = o
    #     else:
    #         o = {}
    #         o['hash'] = h
    #         o['drug_id'] = drugId
    #         o['smiles'] = smiles
    #         o['all_ids'] = [drugId]
    #         df.add()
    #         hashes[h] = o
    # write_to_csv(file, hashes,
    #               ['hash','drug_id','smiles','all_ids'])
    # write_to_csv(f'{output_directory}/opentargets_no_smiles4drugs.csv', invalid_smiles,
    #               ['drug_id'])

    # hashes = {}
    # for targetId in tqdm(target_ids, desc="Processing targets"):
    #     sequence = get_sequence_from_targetId(targetId)
    #     h = hash(sequence)
    #     if h in hashes:
    #         # retrieve the object
    #         o = hashes[h]
    #         if targetId not in o['other_ids']:
    #             o['other_ids'].append(targetId)
    #         hashes[h] = o
    #     else:
    #         o = {}
    #         o['hash'] = h
    #         o['target_id'] = targetId
    #         o['sequence'] = sequence
    #         o['all_ids'] = [targetId]
    #         hashes[h] = o
    # hashes
    # write_to_csv(f'{output_directory}/opentargets_targets.csv',
    #              hashes, ['hash','target_id','sequence','all_ids'])


if __name__ == "__main__":
    target_directory = "data/download/opentargets/knownDrugsAggregated"
    output_directory = "data/processed"  # Replace with desired output CSV file name/path
    os.makedirs(output_directory, exist_ok=True)
    prepare(target_directory, output_directory)
