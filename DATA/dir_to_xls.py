import os
import csv

DATADIRS = ["./segformer_dataset255_all/test",
           "./segformer_dataset255_all/val",
           "./segformer_dataset255_all/train"]

for data_path in DATADIRS:
  data = {"image_path": [], "mask_path": []}

  labs = os.path.join(data_path, 'annotations')
  imgs = os.path.join(data_path, 'images')

  for img in os.listdir(imgs):
    data["image_path"].append(os.path.abspath(os.path.join(imgs, img)))
    data["mask_path"].append(os.path.abspath(os.path.join(labs, img.replace(".jpg", ".png"))))

  print('there are', len(data["image_path"]))

  csv_path = os.path.join(data_path, "index.csv")
  with open(csv_path, 'w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=["image_path", "mask_path"])
      writer.writeheader()
      for i in range(len(data["image_path"])):
          writer.writerow({"image_path": data["image_path"][i], "mask_path": data["mask_path"][i]})

  print('ok', data_path)
