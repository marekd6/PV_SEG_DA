'''
entry point to define and run tuning scenarios
stage 3: DK/DK or GDA
'''

import wandb
import training
import params_synt as args # type: ignore
import types
from torch import cuda
from os.path import join
from random import random


# data paths
test_s = "/users/project1/pt01299/synt/segformer_dataset255_all/test/index.csv"
val_s = "/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv"
train_s = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index"

val_r = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/val/index.csv'
train_r = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/train/index.csv'
test_r =  '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/herlev_test/test/index.csv'

test_gda = '/users/project1/pt01299/synt/gda70/train/index_test.csv'
val_gda = '/users/project1/pt01299/synt/gda70/train/index_val.csv'
val_gda_tr = '/users/project1/pt01299/synt/gda70/train/index_val_tr.csv'
val_gda_val = '/users/project1/pt01299/synt/gda70/train/index_val_val.csv'
full_gda = '/users/project1/pt01299/synt/gda70/train/index.csv'

val_m = "/users/project1/pt01299/synt/mix_val.csv"
train_m = "/users/project1/pt01299/synt/mix_train"


def train3():
    base_settings = {
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }
    print(f"\n--- DEBUG: base_vars has {len(base_settings.keys())} items ---")
    print(base_settings)
    print("---------------------------------------------------\n")

    writer = wandb.init(config=base_settings)

    for key, value in base_settings.items():
        if key not in wandb.config:
            wandb.config[key] = value
    config = wandb.config
    
    print(f"--- Starting Run ---")
    for k, v in config.items():
        print(k, '=', v)
    
    sd = config.save_dir[:-1]
    base_settings.update(dict(wandb.config))
    mod_pt = join(sd, config.mod_ph2, 'model_ph2.pth')

    training.train_model(val_gda_tr, config.val, [test_s, test_r, test_gda], writer, mod_pt, 3, base_settings)

    cuda.empty_cache() # magic


def train1():
    base_settings = {
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }
    print(f"\n--- DEBUG: base_vars has {len(base_settings.keys())} items ---")
    print(base_settings)
    print("---------------------------------------------------\n")

    writer = wandb.init(config=base_settings)

    for key, value in base_settings.items():
        if key not in wandb.config:
            wandb.config[key] = value
    config = wandb.config
    
    print(f"--- Starting Run ---")
    for k, v in config.items():
        print(k, '=', v)
    
    base_settings.update(dict(wandb.config))

    training.train_model(val_gda_tr, config.val, [test_s, test_r, test_gda], writer, None, 1, base_settings)

    cuda.empty_cache() # magic


if __name__ == "__main__":
    if random() > 0.6:
        for _ in range(33):
            wandb.agent("lb/pv_gda/afi2hm6i", function=train1, count=6)
            wandb.agent("lb/pv_gda/xpq8fvpw", function=train3, count=4)
            wandb.agent("lb/pv_gda/5k0muz34", function=train3, count=4)
            wandb.agent("lb/pv_gda/dgd0sx7h", function=train3, count=4)

    else:
        for _ in range(33):
            wandb.agent("lb/pv_gda/dgd0sx7h", function=train3, count=5)
            wandb.agent("lb/pv_gda/afi2hm6i", function=train1, count=3)
            wandb.agent("lb/pv_gda/xpq8fvpw", function=train3, count=5)
            wandb.agent("lb/pv_gda/5k0muz34", function=train3, count=5)
