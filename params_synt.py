'''
adapted from SS
args for compatability
and with const values
'''

# config.py

model_name = "segformer-b5-ready" # segformer-b5, deeplabv3-resnet101", "sam""segformer-b5""unet"
loss = "Dice+BCE"  # Options: ["BCE", "Dice", "Combo", "Focal", "Dice+Focal", "Dice+BCE"]

seed = 0
device = "cuda"  # Options: ['cuda', 'cpu']
max_len = 256
augmentation = True # Whether to use augmentation
optim = "adamw"  # Options: ['adam', 'adamw']
report_to = 'wandb'  # Options: ['wandb', 'tensorboard', None]
writer = '' # ***********************************************************
workers = 12  # Number of dataloader workers
save_dir = "/users/project1/pt01299/synt/fts2"

epochs = 16
iou_decay_fact = 0.7
lr_layer_decay = 0.85
