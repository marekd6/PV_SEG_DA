import training
import params as args # type: ignore

test_s = "/users/project1/pt01299/synt/segformer_dataset255_all/test/index.csv"
val_s = "/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv"
train_s = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index.csv"
train_s = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index_"
tv_s = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index.csv"

test_r =  '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/herlev_test/test/index.csv'
val_r = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/val/index.csv'
train_r = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/train/index.csv'

val_m = "/users/project1/pt01299/synt/mix_val.csv"
train_m = "/users/project1/pt01299/synt/mix_train.csv"
train_m = "/users/project1/pt01299/synt/mix_train_"
subs = ['pnl0.csv', 'pnl1.csv', 'pnl2.csv', 'pnl3.csv', 'pnl4.csv', 'pnl5.csv', 'pnl6.csv', 'pnl7.csv', 'pnl8.csv', 'mix.csv', 'composite.csv']
subs = [str(i)+'.csv' for i in range(15, 80, 10)]

test_gda = '/users/project1/pt01299/synt/gda70/train/index_test.csv'
val_gda = '/users/project1/pt01299/synt/gda70/train/index_val.csv'
full_gda = '/users/project1/pt01299/synt/gda70/train/index.csv'

args.training_ratio=1
args.loss = "Dice+BCE"
args.resume_ckpt = None
args.report_to = None
args.save_dir = "/users/project1/pt01299/synt/fts"
args.model_name = "segformer-b5-ready"

args.optim = "adamw"
args.batch_size = 4
args.exp_date = 78-1
args.exp_date = 150

for sub in subs:

    print('now sub is', sub)

    args.warmup_epochs = 0
    args.wd = 0.1
    args.exp_date = args.exp_date + 1
    args.lr = 1.5e-6
    args.lrenc = 1e-6
    args.lrdec = 1e-5
    args.epochs = 5
    args.exp_code = f'{args.model_name}_{args.exp_date}'
    print('starting', args.exp_code)
    mod_pth = training.train_model(train_s+sub, val_gda, [test_s, test_r, test_gda], None, None, 'ph1', args)
    args.lrenc = 5e-7
    args.lrdec = 5e-6
    args.epochs = 7
    mod_pth = training.train_model(train_m+sub, val_gda, [test_s, test_r, test_gda], None, mod_pth, 'ph2', args)
    args.lrenc = 5e-7
    args.lrdec = 5e-6
    args.epochs = 3
    mod_pth = training.train_model(train_r, val_gda, [test_s, test_r, test_gda], None, mod_pth, 'ph3', args)
    print('DONE', args.exp_code)
    
    args.warmup_epochs = 0
    args.wd = 0.1
    args.exp_date = args.exp_date + 1
    args.lr = 1.5e-6
    args.lrenc = 5e-6
    args.lrdec = 5e-5
    args.epochs = 7
    args.exp_code = f'{args.model_name}_{args.exp_date}'
    print('starting', args.exp_code)
    mod_pth = training.train_model(train_s+sub, val_gda, [test_s, test_r, test_gda], None, None, 'ph1', args)
    args.lrenc = 1e-6
    args.lrdec = 1e-5
    args.epochs = 5
    mod_pth = training.train_model(train_m+sub, val_gda, [test_s, test_r, test_gda], None, mod_pth, 'ph2', args)
    args.lrenc = 5e-7
    args.lrdec = 5e-6
    args.epochs = 3
    mod_pth = training.train_model(train_r, val_gda, [test_s, test_r, test_gda], None, mod_pth, 'ph3', args)
    print('DONE', args.exp_code)
    
    args.warmup_epochs = 5
    args.wd = 0.1
    args.exp_date = args.exp_date + 1
    args.lr = 1.5e-6
    args.lrenc = 5e-6
    args.lrdec = 5e-5
    args.epochs = 7
    args.exp_code = f'{args.model_name}_{args.exp_date}'
    print('starting', args.exp_code)
    mod_pth = training.train_model(train_s+sub, val_gda, [test_s, test_r, test_gda], None, None, 'ph1', args)
    args.lrenc = 1e-6
    args.lrdec = 1e-5
    args.epochs = 5
    mod_pth = training.train_model(train_m+sub, val_gda, [test_s, test_r, test_gda], None, mod_pth, 'ph2', args)
    args.lrenc = 5e-7
    args.lrdec = 5e-6
    args.epochs = 3
    mod_pth = training.train_model(train_r, val_gda, [test_s, test_r, test_gda], None, mod_pth, 'ph3', args)
    print('DONE', args.exp_code)

    args.warmup_epochs = 5
    args.batch_size = 2 #################################
    args.wd = 0.1
    args.exp_date = args.exp_date + 1
    args.lr = 1.5e-6
    args.lrenc = 5e-6
    args.lrdec = 5e-5
    args.epochs = 7
    args.exp_code = f'{args.model_name}_{args.exp_date}'
    print('starting', args.exp_code)
    mod_pth = training.train_model(train_s+sub, val_gda, [test_s, test_r, test_gda], None, None, 'ph1', args)
    args.lrenc = 1e-6
    args.lrdec = 1e-5
    args.epochs = 5
    mod_pth = training.train_model(train_m+sub, val_gda, [test_s, test_r, test_gda], None, mod_pth, 'ph2', args)
    args.lrenc = 5e-7
    args.lrdec = 5e-6
    args.epochs = 3
    mod_pth = training.train_model(train_r, val_gda, [test_s, test_r, test_gda], None, mod_pth, 'ph3', args)
    print('DONE', args.exp_code)

    args.batch_size = 4
    args.warmup_epochs = 0
    args.wd = 0.1
    args.exp_date = args.exp_date + 1
    args.lr = 1.5e-6
    args.lrenc = 1e-6
    args.lrdec = 1e-5
    args.epochs = 5
    args.exp_code = f'{args.model_name}_{args.exp_date}'
    print('starting', args.exp_code)
    mod_pth = training.train_model(train_s+sub, val_gda, [test_s, test_r, test_gda], None, None, 'ph1', args)
    args.lrenc = 5e-7
    args.lrdec = 5e-6
    args.epochs = 7
    mod_pth = training.train_model(train_m+sub, val_gda, [test_s, test_r, test_gda], None, mod_pth, 'ph2', args)
    args.lrenc = 5e-7
    args.lrdec = 5e-6
    args.epochs = 3
    mod_pth = training.train_model(train_r, val_gda, [test_s, test_r, test_gda], None, mod_pth, 'ph3', args)
    print('DONE', args.exp_code)

    print('done with sub', sub)
