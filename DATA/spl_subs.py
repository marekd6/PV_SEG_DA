import csv
import random

def sample_csv(input_file, output_file, percentage):
    """
    Reads a CSV, randomly samples a percentage of its rows, and saves to a new CSV.
    """
    # Ensure percentage is valid
    if not (0 < percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")

    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        
        # Extract the header so it isn't mixed into the random sampling
        try:
            header = next(reader)
        except StopIteration:
            print("The input CSV is empty.")
            return
            
        # Read the rest of the rows into a list
        rows = list(reader)
        
    # Calculate exactly how many rows to keep
    sample_size = int(len(rows) * (percentage / 100.0))
    
    # Randomly select the rows
    sampled_rows = random.sample(rows, sample_size)
    
    # Write the header and the sampled rows to the new file
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(sampled_rows)
        
    print(f"Successfully sampled {sample_size} out of {len(rows)} rows ({percentage}%).")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    INPUT_CSV = '/users/project1/pt01299/synt/segformer_dataset255_all/train/index.csv'
    OUTPUT_CSV = '/users/project1/pt01299/synt/segformer_dataset255_all/train/index_'
    
    for perc in range(15, 80, 10):
        sample_csv(INPUT_CSV, OUTPUT_CSV+str(perc)+'.csv', perc)

    INPUT_CSV = '/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv'
    OUTPUT_CSV = '/users/project1/pt01299/synt/segformer_dataset255_all/val/index_'
    
    for perc in range(15, 80, 10):
        sample_csv(INPUT_CSV, OUTPUT_CSV+str(perc)+'.csv', perc)

    INPUT_CSV = '/users/project1/pt01299/synt/segformer_dataset255_all/test/index.csv'
    OUTPUT_CSV = '/users/project1/pt01299/synt/segformer_dataset255_all/test/index_'
    
    for perc in range(15, 80, 10):
        sample_csv(INPUT_CSV, OUTPUT_CSV+str(perc)+'.csv', perc)
