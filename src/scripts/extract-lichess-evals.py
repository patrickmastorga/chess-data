import csv
import json
import random

# Parameters
input_file = 'datasets/lichess/lichess_db_eval.jsonl'
output_file = 'datasets/lichess_evals_filtered.csv'
overwrite = True
subset_size = 5000000  # Adjust as needed
probability_threshold = subset_size / 22000000 / 0.8  # Approximately 80 percent of the entries have valid cp evals

print("Extracting data")

# Open CSV file in append mode
with open(output_file, 'w' if overwrite else 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header if the file is empty
    if csvfile.tell() == 0:
        writer.writerow(['FEN', 'Evaluation'])

    count = 0

    with open(input_file, 'r') as f:
        # Iterate through the lines
        for line_number, line in enumerate(f):
            # Probabilistically decide whether to include the line
            if random.random() < probability_threshold:
                entry = json.loads(line)

                # Extract required data from the entry
                fen = entry['fen'].split(' ')[0]
                evals = entry['evals']

                # Extract 'cp' entry (if it exists) from the first entry of 'pvs' array in the first entry of 'evals' array
                pvs = evals[0].get('pvs', [])
                cp = pvs[0].get('cp', None) if pvs else None

                # Write data to CSV
                if cp:
                    writer.writerow([fen, cp])
                    count += 1
                    print(count, end='\r')

print("Done!")