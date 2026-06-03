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

val_r = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/val/index.csv'
train_r = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/train/index.csv'
test_r =  '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/herlev_test/test/index.csv'

test_gda = '/users/project1/pt01299/synt/gda70/train/index_test.csv'
val_gda = '/users/project1/pt01299/synt/gda70/train/index_val.csv'
full_gda = '/users/project1/pt01299/synt/gda70/train/index.csv'

val_m = "/users/project1/pt01299/synt/mix_val.csv"
train_m = "/users/project1/pt01299/synt/mix_train"

def train():
    base_settings = {
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }
    print(f"\n--- DEBUG: base_vars has {len(base_settings.keys())} items ---")
    # base_settings["epochs"] = 16
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

    training.train_model(train_s, config.val, [test_s, test_r, test_gda], writer, None, 1, base_settings)

    cuda.empty_cache() # magic


def train10():
    base_settings = {
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }
    print(f"\n--- DEBUG: base_vars has {len(base_settings.keys())} items ---")
    base_settings["epochs"] = 10
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

    training.train_model(train_s, config.val, [test_s, test_r, test_gda], writer, None, 1, base_settings)

    cuda.empty_cache() # magic


def train5():
    base_settings = {
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }
    print(f"\n--- DEBUG: base_vars has {len(base_settings.keys())} items ---")
    base_settings["epochs"] = 5
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

    training.train_model(train_s, config.val, [test_s, test_r, test_gda], writer, None, 1, base_settings)

    cuda.empty_cache() # magic


def train30():
    base_settings = {
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }
    print(f"\n--- DEBUG: base_vars has {len(base_settings.keys())} items ---")
    base_settings["epochs"] = 30
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

    training.train_model(train_s, config.val, [test_s, test_r, test_gda], writer, None, 1, base_settings) # !!!!!!!!!!!!!!!!1

    cuda.empty_cache() # magic


def train100dk():
    base_settings = {
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }
    print(f"\n--- DEBUG: base_vars has {len(base_settings.keys())} items ---")
    base_settings["epochs"] = 100
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

    training.train_model(train_r, config.val, [test_s, test_r, test_gda], writer, None, 1, base_settings) # !!!!!!!!!!!!!!!!1

    cuda.empty_cache() # magic


def traindk():
    base_settings = {
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }
    print(f"\n--- DEBUG: base_vars has {len(base_settings.keys())} items ---")
    # base_settings["epochs"] = 100
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

    training.train_model(train_r, config.val, [test_s, test_r, test_gda], writer, None, 1, base_settings) # !!!!!!!!!!!!!!!!1

    cuda.empty_cache() # magic


def trainmix():
    base_settings = {
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }
    print(f"\n--- DEBUG: base_vars has {len(base_settings.keys())} items ---")
    # base_settings["epochs"] = 100
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

    training.train_model(train_m, config.val, [test_s, test_r, test_gda], writer, None, 1, base_settings) # !!!!!!!!!!!!!!!!1

    cuda.empty_cache() # magic


if __name__ == "__main__":
    # wandb.agent("marekd6-politechnika-gda-ska/pv_short/sbm0h9yh", function=train, count=6)
    # wandb.agent("marekd6-politechnika-gda-ska/pv_short/74u6rgpc", function=train, count=6)
    # wandb.agent("marekd6-politechnika-gda-ska/pv_ft30/elo8kgsy", function=train30, count=6)
    # wandb.agent("marekd6-politechnika-gda-ska/pv_ft30/tz7oe1lc", function=train30, count=6)
    wandb.agent("marekd6-politechnika-gda-ska/pv_subs/8ej2yv2y", function=trainmix, count=123)
    # wandb.agent("marekd6-politechnika-gda-ska/pv_subs/7l3d9j8z", function=train, count=5)
    # wandb.agent("marekd6-politechnika-gda-ska/pv_dk/2d2nocg2", function=train100dk, count=5)
    # wandb.agent("marekd6-politechnika-gda-ska/pv_dk/9wcg8xc5", function=train100dk, count=5)
    # wandb.agent("marekd6-politechnika-gda-ska/pv_dk/59orbtvr", function=traindk, count=5)
    # wandb.agent("marekd6-politechnika-gda-ska/pv_ft30/6xgk97f7", function=train30, count=4)
    # wandb.agent("marekd6-politechnika-gda-ska/pv_ft30/th5r08lg", function=train30, count=2)
