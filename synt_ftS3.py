'''
entry point to define and run tuning scenarios
stage 1: SYNT/SYNT or GDA
'''

import wandb
import training
import params_synt as argsq # type: ignore
import types
from torch import cuda
import argparse


# data paths
test_s = "/users/project1/pt01299/synt/segformer_dataset255_all/test/index.csv"
val_s = "/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv"
train_s = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index"
# train_sv = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index"

test_r =  '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/herlev_test/test/index.csv'

test_gda = '/users/project1/pt01299/synt/gda70/train/index_test.csv'
val_gda = '/users/project1/pt01299/synt/gda70/train/index_val.csv'
full_gda = '/users/project1/pt01299/synt/gda70/train/index.csv'


# params
subs = ['', 'pnl0', 'pnl1', 'pnl2', 'pnl3', 'pnl4', 'pnl5', 'pnl6', 'pnl7', 'pnl8', 'mix', 'composite']
subs = ['mix', 'composite']
subs.extend([str(i) for i in range(15, 80, 10)])

sweep_config = {
    "name": "subs",
    "method": "bayes", 
    "metric": {
        "name": "1_test_GDA_iou",
        "goal": "maximize"
    },
    "parameters": {
        "val": {
            "values": [val_s] # S2
            # "values": [val_gda]
        },
        "sub": {
            "values": subs # S3
            # "values": ['']
        },   
        "batch_size": {
            "values": [4, 8, 16, 32]
        },        
        "warmup_epochs": {
            "values": [4, 2, 6]
        },
        "wd": {
            "values": [0.1, 0.01, 0.05]
        },
        "lrenc": {
            "values": [1e-6, 5e-6, 8e-6, 5e-5]
        },        
        "lrdec": {
            "values": [5e-6, 8e-5, 1e-5, 5e-4]
        },

        "lr_scheduler": {
            "values": ['cosine', 'polynomial']
        },    
        "loss": {
            "values": ["Dice+BCE", "Dice+Focal"]
        }          
    }
}


def train():
    base_settings = {
        key: value for key, value in vars(argsq).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }
    print(f"\n--- DEBUG: base_vars has {len(base_settings.keys())} items ---")
    print(base_settings)
    print("---------------------------------------------------\n")

    writer = wandb.init(config=base_settings)
    # writer = wandb.init()
    # wandb.config.update(base_settings, allow_val_change=False)
    
    for key, value in base_settings.items():
        if key not in wandb.config:
            wandb.config[key] = value
    config = wandb.config
    
    print(f"--- Starting Run ---")
    for k, v in config.items():
        print(k, '=', v)

    base_settings.update(dict(wandb.config))

    training.train_model(train_s, config.val, [test_s, test_r, test_gda], writer, None, 1, base_settings)

    cuda.empty_cache() # magic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    args = parser.parse_args()
    # sweep_id = wandb.sweep(sweep_config, project="pv_ftS3")
    wandb.agent(args.sweep_id, function=train, count=16)
