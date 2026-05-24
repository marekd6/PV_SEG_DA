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
training_ratio = 1 # Training ratio
lr_scheduler = "cosine"  # Options: ['cosine', 'fixed', 'poly']
augmentation = True # Whether to use augmentation
gc = 1  # Gradient accumulation
folds = 0  # Number of folds
prop = 1  # Proportion of training data
optim = "adamw"  # Options: ['adam', 'adamw']
report_to = 'wandb'  # Options: ['wandb', 'tensorboard', None]
writer = '' # ***********************************************************
workers = 12  # Number of dataloader workers
save_dir = "/users/project1/pt01299/synt/fts2"

epochs1 = 12
batch_size1 = 4
warmup_epochs1 = 0
wd1 = 0
lrenc1 = 1e-6
lrdec1 = 1e-5
iou_decay_fact1 = 0.7
