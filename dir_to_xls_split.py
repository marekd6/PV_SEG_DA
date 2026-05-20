import os
import csv
from collections import defaultdict

LAB_EXT = '.png'

def get_group_id(filename: str):
    x = filename.replace(LAB_EXT, '').split('_')[-1]
    if x == 'aug':
      x = filename.replace(LAB_EXT, '').split('_')[-2]
    return x


def save_to_csv(pth, records: list):
  print('there are', len(records))
  with open(pth, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "mask_path"])
    writer.writeheader()
    for lab in records:
      writer.writerow({"image_path": lab.replace(LAB_EXT, ".jpg").replace('annotations', 'images'), "mask_path": lab})
   

DATADIRS = ["./segformer_dataset255_all/test",
           "./segformer_dataset255_all/val",
           "./segformer_dataset255_all/train"]

# DATADIRS = ['/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/herlev_test/test']

# DATADIRS = ['/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/train',
# '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/val']

# DATADIRS = ['/users/project1/pt01299/synt/gda70/train']


for data_path in DATADIRS:
  labs = os.path.join(data_path, 'annotations')
  # labs = os.path.join(data_path, 'labels')
  # imgs = os.path.join(data_path, 'images')
  # labs = os.path.join(data_path, 'mask')
  # imgs = os.path.join(data_path, 'positive')

  groups = defaultdict(list)

  for lab in os.listdir(labs):
    groups[get_group_id(lab)].append(os.path.abspath(os.path.join(labs, lab)))
  
  groups.pop('empty')

  for variant in groups.keys():
    csv_path = os.path.join(data_path, f"index_{variant}.csv")
    save_to_csv(csv_path, groups[variant])
    print(csv_path)

  print('ok', data_path)
