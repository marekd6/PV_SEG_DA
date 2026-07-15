import wandb
import training
import params as args # type: ignore
import types
from torch import cuda

# data paths
test_s = "/users/project1/pt01299/synt/segformer_dataset255_all/test/index.csv"
val_s = "/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv"
train_s = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index"

tv_s = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index.csv"

test_r =  '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/herlev_test/test/index.csv'
val_r = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/val/index.csv'
train_r = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/train/index.csv'

val_m = "/users/project1/pt01299/synt/mix_val.csv"
train_m = "/users/project1/pt01299/synt/mix_train"

test_gda = '/users/project1/pt01299/synt/gda70/train/index_test.csv'
val_gda = '/users/project1/pt01299/synt/gda70/train/index_val.csv'
full_gda = '/users/project1/pt01299/synt/gda70/train/index.csv'


# params
subs = ['pnl0.csv', 'pnl1.csv', 'pnl2.csv', 'pnl3.csv', 'pnl4.csv', 'pnl5.csv', 'pnl6.csv', 'pnl7.csv', 'pnl8.csv', 'mix.csv', 'composite.csv']
subs = ['', 'pnl0', 'pnl1', 'pnl2', 'pnl3', 'pnl4', 'pnl5', 'pnl6', 'pnl7', 'pnl8', 'mix', 'composite']
# subs = [str(i)+'.csv' for i in range(15, 80, 10)]
subs.extend([str(i) for i in range(15, 80, 10)])

sweep_config = {
    # Change "bayes" to "grid" if you want to strictly evaluate every combination
    "name": "basic",
    "method": "bayes", 
    "metric": {
        "name": "3_test_GDA_loss",
        "goal": "minimize"
    },
    "parameters": {
        # Strict scenario (Grid-style choices)
        # "train1": {
        #     "values": [train_s, train_r, train_m]
        # },           
        # "val1": {
        #     "values": [val_s, val_r, val_m, val_gda]
        # },    
        "trains": {
            "values": [
                # train_s + ',' + train_r + ',' + train_m,
                train_s + ',' + train_m + ',' + train_r,
                train_m + ',' + train_s + ',' + train_r,
            ]
        },           
        "vals": {
            "values": [
                val_gda + ',' + val_gda + ',' + val_gda,
                val_r + ',' + val_r + ',' + val_r,
                # val_s + ',' + val_m + ',' + val_r,
                # val_m + ',' + val_s + ',' + val_r,
            ]
        },
        "sub": {
            # "values": subs
            "values": ['']
        },   

        "batch_size1": {
            "values": [2, 4, 8]
        },        
        "warmup_epochs1": {
            "values": [0, 4]
        },
        "wd1": {
            "values": [0, 0.1]
        },
        # Bayesian heuristic (Continuous range)
        "lr1": {
            "distribution": "log_uniform_values",
            "min": 1.5e-7,
            "max": 1.5e-6,
        },        
        "min_lr1": {
            "distribution": "log_uniform_values",
            "min": 1e-8,
            "max": 1.5e-6
        },        
        "lrenc1": {
            "distribution": "log_uniform_values",
            "min": 1e-7,
            "max": 1.5e-6
        },        
        "lrdec1": {
            "distribution": "log_uniform_values",
            "min": 1e-7,
            "max": 1.5e-6
        },        
        
        "batch_size2": {
            "values": [2, 4, 8]
        },        
        "warmup_epochs2": {
            "values": [0, 4]
        },
        "wd2": {
            "values": [0, 0.1]
        },
        # Bayesian heuristic (Continuous range)
        "lr2": {
            "distribution": "log_uniform_values",
            "min": 1.5e-7,
            "max": 1.5e-6
        },        
        "min_lr2": {
            "distribution": "log_uniform_values",
            "min": 5e-8,
            "max": 1.5e-6
        },        
        "lrenc2": {
            "distribution": "log_uniform_values",
            "min": 1e-7,
            "max": 1.5e-6
        },        
        "lrdec2": {
            "distribution": "log_uniform_values",
            "min": 1e-7,
            "max": 1.5e-6
        },    

        "batch_size3": {
            "values": [2, 4, 8]
        },        
        "warmup_epochs3": {
            "values": [0, 4]
        },
        "wd3": {
            "values": [0, 0.1]
        },
        # Bayesian heuristic (Continuous range)
        "lr3": {
            "distribution": "log_uniform_values",
            "min": 1.5e-7,
            "max": 1.5e-6
        },        
        "min_lr3": {
            "distribution": "log_uniform_values",
            "min": 5e-8,
            "max": 1.5e-6
        },        
        "lrenc3": {
            "distribution": "log_uniform_values",
            "min": 1e-7,
            "max": 1.5e-6
        },        
        "lrdec3": {
            "distribution": "log_uniform_values",
            "min": 1e-7,
            "max": 1.5e-6
        },
    }
}

# ==========================================
# 2. DEFINE THE MAIN TRAINING FUNCTION
# ==========================================
def train():
    # Convert your imported 'params.py' module to a dictionary safely
    base_settings = {
        key: value for key, value in vars(args).items() 
        if not key.startswith('__') and not isinstance(value, types.ModuleType)
    }

    # Initialize the run. 
    # W&B automatically overwrites these base_settings with the specific 
    # hyperparameters chosen by the Sweep Controller for this iteration.
    writer = wandb.init(config=base_settings)

    # Fetch the final merged configuration
    config = wandb.config

    # config.exp_code = writer.id
    
    print(f"--- Starting Run ---")
    # print(f"Model: {config.model_name} | Dataset: {config.dataset_name}")
    # print(f"LR: {config.learning_rate} | Batch: {config.batch_size} | Opt: {config.optimizer}")
    for k, v in config.items():
        print(k, '=', v)
    
    trains = config.trains.split(',')
    vals = config.vals.split(',')

    # mod_pth = training.train_model(config.train1, config.val1, [test_s, test_r, test_gda], config.writer, None, 1, config)
    mod_pth = training.train_model(trains[0], vals[0], [test_s, test_r, test_gda], writer, None, 1, config)
    mod_pth = training.train_model(trains[1], vals[1], [test_s, test_r, test_gda], writer, mod_pth, 2, config)
    mod_pth = training.train_model(trains[2], vals[2], [test_s, test_r, test_gda], writer, mod_pth, 3, config)

    cuda.empty_cache() # magic

# ==========================================
# 3. LAUNCH THE SWEEP AGENT
# ==========================================
if __name__ == "__main__":
    # Create the Sweep Controller on W&B's servers and get the ID
    # Note: If running on SLURM, you typically run this line once on your laptop, 
    # and pass the resulting sweep_id directly into wandb.agent() in your SLURM script.
    sweep_id = wandb.sweep(sweep_config, project="pv_da")
    # sweep_id = 'f7qcq2uh'
    
    # Start the agent to execute the runs locally.
    # count=10 tells the agent to run 10 different parameter combinations before stopping.
    wandb.agent(sweep_id, function=train, count=12)

# sub potem albo jako baza na początek (60%?)
