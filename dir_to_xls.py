import os
import csv

DATADIR="./segformer_dataset/test"

data = {"image_path": [], "mask_path": []}

data_path = DATADIR

labs = os.path.join(DATADIR, 'annotations')
imgs = os.path.join(DATADIR, 'images')

for img in os.listdir(imgs):
  data["image_path"].append(os.path.abspath(os.path.join(imgs, img)))
  data["mask_path"].append(os.path.abspath(os.path.join(labs, img.replace(".jpg", ".png"))))

print('there are', len(data["image_path"]))

csv_path = os.path.join(DATADIR, "test_abs.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "mask_path"])
    # writer.writeheader()
    for i in range(len(data["image_path"])):
        writer.writerow({"image_path": data["image_path"][i], "mask_path": data["mask_path"][i]})

print('ok')
