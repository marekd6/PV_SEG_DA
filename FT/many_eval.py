import training, data, models
from torch.utils.data import DataLoader
import params as args # type: ignore
# import pred
# import pandas as pd
import os
import glob
import torch


test_path = "/users/project1/pt01299/synt/gda/test/test.csv"
# image_size = 625
# mask_size = 625

# args.batch_size = 4
# args.lr = 0.00002
# args.training_ratio=1
# args.optim = "adam"
# args.loss = "Dice+BCE"
# args.resume_ckpt ="/content/drive/MyDrive/solar_PV_prediction/outputs_segAug/segformer-b5_25-06-20_lr_2e-05_opt_adam_bsz_4_ratio_1_loss_Dice+BCE_country_China/model_39.pth"
args.model_name = "segformer-b5"# deeplabv3-resnet101", "sam""segformer-b5""unet"
# args.dataset_csv = "/content/drive/MyDrive/solar_PV_prediction/src/dataset_csv"
args.report_to = None

model_vers = glob.glob(os.path.join('/users/project1/pt01299/synt/fts', "**/*.pth"))
print('evaluation on', test_path)
for chk_pt in model_vers:
    print(chk_pt)
    model, processor, image_size, mask_size = models.load_model(args.model_name, args.device)
    model.load_state_dict(torch.load(chk_pt))
    criterion = training.get_criterion(args)
    test_data = data.SegmentationDataset(test_path, image_size=image_size, mask_size=mask_size, transform=processor, model_name=args.model_name)
    test_dl = DataLoader(test_data, batch_size=16, num_workers=args.workers, shuffle=False)
    val_loss, val_stats = training.eval_one_epoch(model, test_dl, criterion, 1, None, image_size, args)
    print('print', chk_pt)
    print(val_stats)
    print('printed', chk_pt)

# test_csv = pd.read_csv(test_path)
# test_images = test_csv["image_path"].tolist()

# raw_images, pred_masks = run_prediction(MODELPATH, CLASSIFIERPATH, CLASSIFIERNAME, IMGDIR, SAVE_DIR, batch_size=batch_size, model_name=MODELNAME)
#pred.run_prediction('/users/project1/pt01299/synt/solar_scope_md/src/xSegformer-B5.pth', '', '', '/users/project1/pt01299/synt/segformer_dataset/test/images', '/users/project1/pt01299/synt/preds255', args.model_name)
#pred.run_prediction('/users/project1/pt01299/synt/solar_scope_md/src/xSegformer-B5.pth', '', '', '/users/project1/pt01299/synt/bez_cienia', '/users/project1/pt01299/synt/preds255bc2', args.model_name)
