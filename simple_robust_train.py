import training
import params as args # type: ignore


test_s = "/users/project1/pt01299/synt/segformer_dataset255_all/test/index.csv"
val_s = "/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv"
train_s = "/users/project1/pt01299/synt/segformer_dataset255_all/train/index.csv"

test_r =  '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/herlev_test/test/index.csv'
val_r = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/val/index.csv'
train_r = '/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/train/index.csv'

val_m = "/users/project1/pt01299/synt/mix_val.csv"
train_m = "/users/project1/pt01299/synt/mix_train.csv"

args.batch_size = 8
args.lr = 0.00002
args.training_ratio=1
args.optim = "adamw"
args.loss = "Dice+BCE"
args.resume_ckpt = None
args.model_name = "segformer-b5-ready"
args.report_to = None
args.epochs = 11
args.save_dir = "/users/project1/pt01299/synt/fts"
args.exp_date = 3
args.exp_code = f'{args.model_name}_{args.exp_date}_lr_{args.lr}_opt_{args.optim}_bsz_{args.batch_size}_ratio_{args.training_ratio}_loss_{args.loss}_synt'
print(args)

# model, processor, image_size, mask_size = models.load_model(args.model_name, args.device)

# criterion = training.get_criterion(args)

# test_data = data.SegmentationDataset(test_path, image_size=image_size, mask_size=mask_size, transform=processor, model_name=args.model_name)
# test_dl = DataLoader(test_data, batch_size=8, num_workers=args.workers, shuffle=False)

# val_data = data.SegmentationDataset(val_path, image_size=image_size, mask_size=mask_size, transform=processor, model_name=args.model_name)
# val_dl = DataLoader(val_data, batch_size=8, num_workers=args.workers, shuffle=False)

# train_data = data.SegmentationDataset(train_path, image_size=image_size, mask_size=mask_size, transform=processor, model_name=args.model_name)
# train_dl = DataLoader(train_data, batch_size=8, num_workers=args.workers, shuffle=False)

# val_loss, val_stats = training.eval_one_epoch(model, test_dl, criterion, 1, None, image_size, args)
# print(val_stats)


args.lr = 0.00004
mod_pth = training.train_model(train_s, val_s, [test_s, test_r], None, None, 'ph1', args)
args.lr = 0.00003
mod_pth = training.train_model(train_m, val_m, [test_s, test_r], None, mod_pth, 'ph2', args)
args.lr = 0.00002
mod_pth = training.train_model(train_r, val_r, [test_s, test_r], None, mod_pth, 'ph3', args)



# test_csv = pd.read_csv(test_path)
# test_images = test_csv["image_path"].tolist()


# raw_images, pred_masks = run_prediction(MODELPATH, CLASSIFIERPATH, CLASSIFIERNAME, IMGDIR, SAVE_DIR, batch_size=batch_size, model_name=MODELNAME)
#pred.run_prediction('/users/project1/pt01299/synt/solar_scope_md/src/xSegformer-B5.pth', '', '', '/users/project1/pt01299/synt/segformer_dataset/test/images', '/users/project1/pt01299/synt/preds255', args.model_name)
#pred.run_prediction('/users/project1/pt01299/synt/solar_scope_md/src/xSegformer-B5.pth', '', '', '/users/project1/pt01299/synt/bez_cienia', '/users/project1/pt01299/synt/preds255bc2', args.model_name)
