import os

data_path = '/users/project1/pt01299/synt/gda70/train'

with open('/users/project1/pt01299/synt/gda70/train/index.csv', 'r', newline='') as full_gda:
  full_data = full_gda.read().splitlines(True)

csv_path = os.path.join(data_path, "index_test.csv")
with open(csv_path, 'w') as f:
  for i in range(50):
    f.write(full_data[i])

csv_path = os.path.join(data_path, "index_val.csv")
with open(csv_path, 'w') as f:
  f.write("image_path,mask_path\n")
  for i in range(50, 70):
    f.write(full_data[i])

  print('ok', data_path)
