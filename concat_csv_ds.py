'''
concatenate csvs to create mixed ds
'''

import csv


def concatenate_csvs(file1_path, file2_path, output_path):
    with open(file1_path, 'r', newline='') as f1, \
         open(file2_path, 'r', newline='') as f2, \
         open(output_path, 'w', newline='') as out_file:
        
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(out_file)

        header = next(reader1)
        writer.writerow(header)
            
        for row in reader1:
          writer.writerow(row)

        next(reader2)
            
        for row in reader2:
          writer.writerow(row)


synt = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index.csv"
rzecz = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/train/index.csv'
joint = "/users/project1/pt01299/synt/mix_train.csv"

concatenate_csvs(synt, rzecz, joint)


synt = "/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv"
rzecz = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/val/index.csv'
joint = "/users/project1/pt01299/synt/mix_val.csv"

concatenate_csvs(synt, rzecz, joint)

print("CSV files concatenated successfully!")
