'''
adapted from SS
args for compatability
and with const values
'''

# config.py

model_name = "segformer-b5-ready" # segformer-b5, deeplabv3-resnet101", "sam""segformer-b5""unet"
loss = "Dice+BCE"  # Options: ["BCE", "Dice", "Combo", "Focal", "Dice+Focal", "Dice+BCE"]

# exp_date = 12345
# exp_code = 741741
seed = 0
device = "cuda"  # Options: ['cuda', 'cpu']
max_len = 256
training_ratio = 1 # Training ratio
lr_scheduler = "cosine"  # Options: ['cosine', 'fixed']
augmentation = True # Whether to use augmentation
gc = 1  # Gradient accumulation
folds = 0  # Number of folds
prop = 1  # Proportion of training data
optim = "adamw"  # Options: ['adam', 'adamw']
# wd = 0  # Weight decay
report_to = 'wandb'  # Options: ['wandb', 'tensorboard', None]
writer = '' # ***********************************************************
workers = 12  # Number of dataloader workers
# resume_ckpt = None
save_dir = "/users/project1/pt01299/synt/fts2"

lr = 2e-4  # Learning rate
min_lr = 1e-6  # Minimum learning rate !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
lrenc = 1e-6
lrdec = 1e-5

# testGDA = ''
# testSYNT = ''
# testDK = ''

eps_save = -1

trains = ''
vals = ''

epochs1 = 7
# train1 = ''
# val1 = ''
batch_size1 = 4
warmup_epochs1 = 0
wd1 = 0
optim1 = "adamw"
lr1 = 2e-4  # Learning rate
min_lr1 = 1e-6  # Minimum learning rate !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
lrenc1 = 1e-6
lrdec1 = 1e-5
iou_decay_fact1 = 0.8

epochs2 = 5
# train2 = ''
# val2 = ''
batch_size2 = 4
warmup_epochs2 = 0
wd2 = 0
optim2 = "adamw"
lr2 = 2e-4  # Learning rate
min_lr2 = 1e-6  # Minimum learning rate !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
lrenc2 = 1e-6
lrdec2 = 1e-5
iou_decay_fact2 = 0.8

epochs3 = 3
# train3 = ''
# val3 = ''
batch_size3 = 4
warmup_epochs3 = 0
wd3 = 0
optim3 = "adamw"
lr3 = 2e-4  # Learning rate
min_lr3 = 1e-6  # Minimum learning rate !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
lrenc3 = 1e-6
lrdec3 = 1e-5
iou_decay_fact3 = 0.8
