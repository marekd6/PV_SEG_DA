'''
entry point to define and run tuning scenarios
stage 1: SYNT/SYNT or GDA
'''

import wandb


# data paths
test_s = "/users/project1/pt01299/synt/segformer_dataset255_all/test/index.csv"
val_s = "/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv"
train_s = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index"

test_r =  '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/herlev_test/test/index.csv'

test_gda = '/users/project1/pt01299/synt/gda70/train/index_test.csv'
val_gda = '/users/project1/pt01299/synt/gda70/train/index_val.csv'
full_gda = '/users/project1/pt01299/synt/gda70/train/index.csv'


# params
subs = ['', 'pnl0', 'pnl1', 'pnl2', 'pnl3', 'pnl4', 'pnl5', 'pnl6', 'pnl7', 'pnl8', 'mix', 'composite']
subs = ['mix', 'composite']
subs.extend([str(i) for i in range(15, 80, 10)])

fixed_subs = {
    "name": "fixed_subs16A",
    "method": "grid", 
    "metric": {
        "name": "1_test_GDA_iou",
        "goal": "maximize"
    },
    "parameters": {
        "val": {
            "values": [val_gda, val_s]
        },
        "sub": {
            "values": subs
        },   
        "batch_size": {
            "values": [16, 8]
        },        
        "warmup_epochs": {
            "values": [6]
        },
        "wd": {
            "values": [0.05]
        },
        "lrenc": {
            "values": [5e-5]
        },        
        "lrdec": {
            "values": [8e-6]
        },

        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+Focal"]
        }          
    }
}

big_srch_gda = {
    "name": "big_srch30gdaA",
    "method": "bayes", 
    "metric": {
        "name": "1_test_GDA_iou",
        "goal": "maximize"
    },
    "parameters": {
        "val": {
            "values": [val_gda]
        },
        "sub": {
            "values": ['']
        },   
        "batch_size": {
            "values": [16, 32]
        },        
        "warmup_epochs": {
            "values": [1]
        },
        "wd": {
            "values": [0.05, 0.01]
        },
        "lrenc": {
            "values": [8e-6, 5e-5, 2e-5]
        },        
        "lrdec": {
            "values": [5e-6, 1e-5]
        },

        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+BCE", "Dice+Focal"]
        }          
    }
}

big_srch_s = {
    "name": "big_srch30sA",
    "method": "bayes", 
    "metric": {
        "name": "1_test_GDA_iou",
        "goal": "maximize"
    },
    "parameters": {
        "val": {
            "values": [val_s]
        },
        "sub": {
            "values": ['']
        },   
        "batch_size": {
            "values": [16, 32]
        },        
        "warmup_epochs": {
            "values": [1]
        },
        "wd": {
            "values": [0.05, 0.01]
        },
        "lrenc": {
            "values": [8e-6, 5e-5, 2e-5]
        },        
        "lrdec": {
            "values": [5e-6, 1e-5]
        },

        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+BCE", "Dice+Focal"]
        }          
    }
}


if __name__ == "__main__":
    sweep_id = wandb.sweep(fixed_subs, project="pv_subs")
    print(sweep_id)
    sweep_id = wandb.sweep(big_srch_gda, project="pv_ft30")
    print(sweep_id)
    sweep_id = wandb.sweep(big_srch_s, project="pv_ft30")
    print(sweep_id)
