'''
entry point to define and run tuning scenarios
stage 1: SYNT/SYNT or GDA
'''

import wandb
import training
import params_synt as args # type: ignore
import types
from torch import cuda

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
subs.extend([str(i) for i in range(15, 80, 10)])

sweep_config = {
    "name": "basic",
    "method": "bayes", 
    "metric": {
        "name": "1_test_GDA_iou",
        "goal": "maximize"
    },
    "parameters": {
        "val": {
            # "values": [val_s] # S2
            "values": [val_gda]
        },
        "sub": {
            # "values": subs # S3
            "values": ['']
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
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }

    writer = wandb.init(config=base_settings)
    config = wandb.config
    
    print(f"--- Starting Run ---")
    for k, v in config.items():
        print(k, '=', v)

    training.train_model(train_s, config.val, [test_s, test_r, test_gda], writer, None, 1, config)

    cuda.empty_cache() # magic


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="pv_ftS1")
    wandb.agent(sweep_id, function=train, count=16)
