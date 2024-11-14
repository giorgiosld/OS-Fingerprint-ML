import csv
import json
import os
import subprocess

# Paths to the files
META_INFO_FILE = '../meta_info_new.csv'
MAP_DUMP_PGD_FILE = '../map-dump-pgd.json'
DUMP_DIRECTORY = '../'
ELF_PTRSCAN_SCRIPT = '../preproc-ml/elf-ptrfeatures.py'

OUTPUT_DATASET_FILE = 'dataset_only_length.csv'

NUM_ROWS = 20

def read_meta_info(file_path):
    """
    Reads the meta info CSV file and extracts entries labeled 'ubuntu,3.13.0' or 'ubuntu,4.4.0'.
    Limits to 10 entries for each label.
    """
    meta_data = []
    count_3_13 = 0
    count_4_4 = 0
    
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader) 
        for row in csv_reader:
            if len(row) > 0:
                dump_hash = row[0]  
                version = row[4].split('-')[0]  # Extract version number before the dash
                label = f"{row[3]},{version}"  
                
                # Limit to 10 entries for each version label
                if label == 'ubuntu,3.13.0' and count_3_13 < 10:
                    meta_data.append({"dump_hash": dump_hash, "label": label})
                    count_3_13 += 1
                elif label == 'ubuntu,4.4.0' and count_4_4 < 10:
                    meta_data.append({"dump_hash": dump_hash, "label": label})
                    count_4_4 += 1
                
                # Stop if both counts reach 10
                if count_3_13 >= 10 and count_4_4 >= 10:
                    break
    
    return meta_data

def get_pgd(dump_hash):
    """
    Retrieves the PGD value for a given dump hash from the PGD JSON map file.
    """
    with open(MAP_DUMP_PGD_FILE, 'r') as jsonfile:
        pgd_map = json.load(jsonfile)
        return pgd_map.get(dump_hash)

def filter_and_add_pgd(meta_info):
    """
    Filters the meta info list to include only entries with a valid PGD.
    If necessary, continues adding more entries to fulfill the requirement of 10 entries per label.
    """
    filtered_meta_info = []
    count_3_13 = 0
    count_4_4 = 0

    # Add PGD values to filtered entries
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
                # Only add if dump hash is not already present
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

    return filtered_meta_info

def run_elf_ptrscan(entry):
    """
    Runs the elf-ptrfeatures.py script to get the length of the pointer graph.
    Extracts and returns the number after 'None,' which represents the length.
    Handles multiple lines of output to ensure all relevant data is captured.
    """
    dump_file = os.path.join(DUMP_DIRECTORY, f"{entry['dump_hash']}.dump")
    pointer_lengths = []  
    try:
        # Run the subprocess to execute the ELF pointer scanning script
        result = subprocess.check_output([
            'python', ELF_PTRSCAN_SCRIPT, '--pgd', entry['pgd'], dump_file
        ], stderr=subprocess.STDOUT)
        output_lines = result.decode('utf-8').splitlines()  
        # Iterate over each line to find lines that start with '[None,'
        for line in output_lines:
            if line.startswith('[None,'):
                parts = line.split(',')
                if len(parts) > 1:
                    pointer_length = int(parts[1].strip())  # Extract the length after 'None,'
                    pointer_lengths.append(pointer_length)  
        return pointer_lengths  
    except subprocess.CalledProcessError as e:
        print(f"Error running ELF script for {entry['dump_hash']}: {e.output.decode('utf-8')}.")
    return []  

def save_to_csv(filtered_meta_info):
    """
    Saves the filtered meta info including pointer graph lengths to a CSV file.
    """
    with open(OUTPUT_DATASET_FILE, 'w', newline='') as csvfile:
        fieldnames = ['dump_hash', 'label', 'pgd', 'pointer_graph_lengths']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in filtered_meta_info:
            pointer_graph_lengths = run_elf_ptrscan(entry)  
            writer.writerow({
                'dump_hash': entry['dump_hash'],
                'label': entry['label'],
                'pgd': entry['pgd'],
                'pointer_graph_lengths': pointer_graph_lengths
            })

def print_filtered_meta_info(filtered_meta_info):
    """
    Prints the filtered meta info list including the pointer graph lengths for all entries.
    """
    for entry in filtered_meta_info:
        pointer_graph_lengths = run_elf_ptrscan(entry) 
        print(f"Dump Hash: {entry['dump_hash']}, Label: {entry['label']}, PGD: {entry['pgd']}, Pointer Graph Lengths: {pointer_graph_lengths}")

if __name__ == "__main__":
    meta_info = read_meta_info(META_INFO_FILE)
    filtered_meta_info = filter_and_add_pgd(meta_info)
    save_to_csv(filtered_meta_info)  
    #print_filtered_meta_info(filtered_meta_info)  

