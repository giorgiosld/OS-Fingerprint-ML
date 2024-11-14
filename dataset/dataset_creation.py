import csv
import json
import subprocess
import os

# Paths to the files
META_INFO_FILE = '../meta_info_new.csv'
MAP_DUMP_PGD_FILE = '../map-dump-pgd.json'
DUMP_DIRECTORY = '../'
ELF_PTRSCAN_SCRIPT = '../preproc-ml/elf-ptrfeatures.py'

# Output dataset file
OUTPUT_DATASET_FILE = 'dataset.csv'

# Number of rows we need
NUM_ROWS = 20

def read_meta_info(file_path):
    meta_data = []
    count_3_13 = 0
    count_4_4 = 0
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            if len(row) > 0:
                dump_hash = row[0]
                # Extract only the version before the dash
                version = row[4].split('-')[0]
                label = f"{row[3]},{version}"
                if label == 'ubuntu,3.13.0' and count_3_13 < 10:
                    meta_data.append({"dump_hash": dump_hash, "label": label})
                    count_3_13 += 1
                elif label == 'ubuntu,4.4.0' and count_4_4 < 10:
                    meta_data.append({"dump_hash": dump_hash, "label": label})
                    count_4_4 += 1
                if count_3_13 >= 10 and count_4_4 >= 10:
                    break
    return meta_data

def get_pgd(dump_hash):
    with open(MAP_DUMP_PGD_FILE, 'r') as jsonfile:
        pgd_map = json.load(jsonfile)
        if dump_hash in pgd_map:
            return pgd_map[dump_hash]
        else:
            return None

if __name__ == "__main__":
    meta_info = read_meta_info(META_INFO_FILE)
    filtered_meta_info = []
    count_3_13 = 0
    count_4_4 = 0

    # Filter entries and add PGD values
    for entry in meta_info:
        pgd = get_pgd(entry['dump_hash'])
        if pgd:
            entry['pgd'] = pgd
            filtered_meta_info.append(entry)
            if entry['label'] == 'ubuntu,3.13.0':
                count_3_13 += 1
            elif entry['label'] == 'ubuntu,4.4.0':
                count_4_4 += 1

    # If we need more rows, continue adding entries according to specific labels
    with open(META_INFO_FILE, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            if len(row) > 0:
                dump_hash = row[0]
                version = row[4].split('-')[0]
                label = f"{row[3]},{version}"
                if dump_hash not in [entry['dump_hash'] for entry in filtered_meta_info]:
                    pgd = get_pgd(dump_hash)
                    if pgd:
                        if label == 'ubuntu,3.13.0' and count_3_13 < 10:
                            filtered_meta_info.append({"dump_hash": dump_hash, "label": label, "pgd": pgd})
                            count_3_13 += 1
                        elif label == 'ubuntu,4.4.0' and count_4_4 < 10:
                            filtered_meta_info.append({"dump_hash": dump_hash, "label": label, "pgd": pgd})
                            count_4_4 += 1
                        if count_3_13 >= 10 and count_4_4 >= 10:
                            break

    # Print the final list
    for entry in filtered_meta_info:
        print(f"Dump Hash: {entry['dump_hash']}, Label: {entry['label']}, PGD: {entry['pgd']}")

