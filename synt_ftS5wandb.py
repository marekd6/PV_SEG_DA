'''
entry point to define and run tuning scenarios
stage 1 and 3
'''

import wandb


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

subs16v2 = {
    "name": "subs16v2",
    "method": "grid", 
    "metric": {
        "name": "1_test_GDA_iou",
        "goal": "maximize"
    },
    "parameters": {
        "val": {
            "values": [val_gda]
        },
        "sub": {
            "values": subs
        },   
        "batch_size": {
            "values": [16, 8]
        },        
        "warmup_epochs": {
            "values": [4]
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


full_scaled5 = {
    "name": "full_scaled5v2",
    "method": "bayes", 
    "metric": {
        "name": "1_test/GDA/iou",
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
            "values": [16]
        },        
        "warmup_epochs": {
            "values": [1, 2]
        },
        "wd": {
            "values": [0.01]
        },
        "lrenc": {
            "values": [5e-5, 5e-6]
        },        
        "lrdec": {
            "values": [8e-6, 5e-6, 1e-5, 5e-5]
        },

        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+BCE", "Dice+Focal"]
        }          
    }
}

full_scaled10 = {
    "name": "full_scaled10v2",
    "method": "bayes", 
    "metric": {
        "name": "1_test/GDA/iou",
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
            "values": [32]
        },        
        "warmup_epochs": {
            "values": [2, 3]
        },
        "wd": {
            "values": [0.03]
        },
        "lrenc": {
            "values": [5e-5, 5e-6]
        },        
        "lrdec": {
            "values": [8e-6, 5e-6, 1e-5, 5e-5]
        },

        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+BCE", "Dice+Focal"]
        }          
    }
}

dk = {
    "name": "dk",
    "method": "bayes", 
    "metric": {
        "name": "1_test/GDA/iou",
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
            "values": [16, 8]
        },        
        "warmup_epochs": {
            "values": [2, 16, 32]
        },
        "wd": {
            "values": [0.03, 0.1, 0.3]
        },
        "lrenc": {
            "values": [5e-5, 1e-5, 5e-6]
        },        
        "lrdec": {
            "values": [8e-6, 5e-6, 1e-5]
        },

        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+BCE", "Dice+Focal"]
        }          
    }
}

full_scaled_eps = {
    "name": "full_scaled_eps2",
    "method": "bayes", 
    "metric": {
        "name": "1_test/GDA/iou",
        "goal": "maximize"
    },
    "parameters": {
        "ema": {
            "values": [False]
        },
        "epochs": {
            "values": [2, 3, 4]
        },
        "val": {
            "values": [val_gda, val_s]
        },
        "sub": {
            "values": ['']
        },   
        "batch_size": {
            "values": [16, 8, 4]
        },        
        "warmup_epochs": {
            "values": [1, 0]
        },
        "wd": {
            "values": [0.02, 0.025, 0.03, 0.01, 0.015, 0.005]
        },
        "lrenc": {
            "values": [5e-6, 2.5e-6, 1e-5, 8e-6]
        },        
        "lrdec": {
            "values": [8e-6, 4e-6, 2e-6, 1e-5]
        },

        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+Focal"]
        }          
    }
}


gda_ph3_subm_m = {
    "name": "gda_ph3_subm_m",
    "method": "bayes", 
    "metric": {
        "name": "3_test/GDA/iou",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {
            "values": [16, 4, 8]
        },
        "mod_ph2": {
            "values": ['gszuy9jw', '6ztbfy35', 'cvu42atm', 'ch9clygy', 'naoka7h9', 'g1reckv0', 'x4o90uoz', 'y57ez3vn', '5u0z2k7k', 'socmlxfg', 'i2gf29uk', 'xwft58d9', 'twr19lwo', 'otnrgdz3', 'wi2jh23x', 'q3l05vl5', 'hdakrj8t', 'xrii4c6m']
        },
        "val": {
            "values": [val_gda_val, val_r]
        },
        "sub": {
            "values": ['']
        },   
        "batch_size": {
            "values": [8, 4, 2]
        },        
        "warmup_epochs": {
            "values": [0, 4]
        },
        "wd": {
            "values": [0.03, 0.015]
        },
        "lrenc": {
            "values": [5e-5]
        },        
        "lrdec": {
            "values": [5e-6]
        },
        "ema":{
            "values": [True, False]
        },
        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+Focal"]
        }          
    }
}


gda_ph3_subm_s = {
    "name": "gda_ph3_subm_s",
    "method": "bayes", 
    "metric": {
        "name": "3_test/GDA/iou",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {
            "values": [16, 4, 8]
        },
        "mod_ph2": {
            "values": ['hf9g8e4w', '2s1u2vh8', 'dxpqn25i', 'nuns2l7q', 'rju0votv', '68e5wqwh', 'nl2464z6', 'c49om9fj', 'ndbecwd2', 'npy2u5kc', '7lp2gz2e', 'm4kiorvc', 'an0q41hy', 'a871n5d0', 'vymtls23', 'ml1iin2r', 'ax37mj8u']
        },
        "val": {
            "values": [val_gda_val, val_r]
        },
        "sub": {
            "values": ['']
        },   
        "batch_size": {
            "values": [8, 4, 2]
        },        
        "warmup_epochs": {
            "values": [0, 4]
        },
        "wd": {
            "values": [0.03, 0.015]
        },
        "lrenc": {
            "values": [5e-5]
        },        
        "lrdec": {
            "values": [5e-6]
        },
        "ema":{
            "values": [True, False]
        },
        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+Focal"]
        }          
    }
}


gda_ph3_subm_gda = {
    "name": "gda_ph3_subm_gda",
    "method": "bayes", 
    "metric": {
        "name": "3_test/GDA/iou",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {
            "values": [16, 4, 8]
        },
        "mod_ph2": {
            "values": ['i6m0fdau', 'xoahfvum', 'qfqiprmf', 'z635glkj', 'tei7bjrr', 'wh1ns2uw', 'ketuajhh', '8tjscjld', 'jadds8mi', '7uxllt9w', 'jlr59yxd', 'mihucmsb', '9r8pockx', 'd5osit6k', 'pevtyaco', 'rkltfhch', 'uafbq2g8']
        },
        "val": {
            "values": [val_gda_val, val_r]
        },
        "sub": {
            "values": ['']
        },   
        "batch_size": {
            "values": [8, 4, 2]
        },        
        "warmup_epochs": {
            "values": [0, 4]
        },
        "wd": {
            "values": [0.03, 0.015]
        },
        "lrenc": {
            "values": [5e-5]
        },        
        "lrdec": {
            "values": [5e-6]
        },
        "ema":{
            "values": [True, False]
        },
        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+Focal"]
        }          
    }
}


gda_ph1 = {
    "name": "gda_ph1",
    "method": "bayes", 
    "metric": {
        "name": "1_test/GDA/iou",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {
            "values": [16, 4, 8, 32]
        },
        "val": {
            "values": [val_gda_val, val_r]
        },
        "sub": {
            "values": ['']
        },   
        "batch_size": {
            "values": [8, 4, 2]
        },        
        "warmup_epochs": {
            "values": [0, 4]
        },
        "wd": {
            "values": [0.03, 0.015]
        },
        "lrenc": {
            "values": [5e-5]
        },        
        "lrdec": {
            "values": [5e-6]
        },
        "ema":{
            "values": [True, False]
        },
        "lr_scheduler": {
            "values": ['cosine']
        },    
        "loss": {
            "values": ["Dice+Focal"]
        }          
    }
}



if __name__ == "__main__":
    sweep_id = wandb.sweep(gda_ph1, project="pv_gda")
    print(sweep_id)
    sweep_id = wandb.sweep(gda_ph3_subm_m, project="pv_gda")
    print(sweep_id)
    sweep_id = wandb.sweep(gda_ph3_subm_s, project="pv_gda")
    print(sweep_id)
    sweep_id = wandb.sweep(gda_ph3_subm_gda, project="pv_gda")
    print(sweep_id)
