'''
training script for SegFormer
adapted from SolarScope
'''

import os
# import cv2
import numpy as np
import wandb
import torch
import data, models, losses, metrics # type: ignore
from transformers import get_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import argparse

GPU_BS = 16 # H100

def get_criterion(args):
    criteria = {
        "bce": losses.BCECriterion(),
        "dice": losses.DiceCriterion(),
        "focal": losses.FocalCriterion(),
        "mse": losses.MSECriterion(),
        "dice+bce": losses.ComboDiceBCECriterion(),
        "dice+focal": losses.ComboDiceFocalCriterion(),
    }
    assert args.loss.lower() in criteria, f"Unknown loss type {args.loss}."
    criterion = criteria[args.loss.lower()]
    return criterion


def get_optimiser(model, encoder_lr, decoder_lr, weight_decay, llrd_decay_rate):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "decode_head" in name:
            layer_id, layer_name = 5, "decoder"         
        elif "encoder.patch_embeddings." in name:
            stage_idx = int(name.split("encoder.patch_embeddings.")[1].split(".")[0])
            layer_id, layer_name = stage_idx + 1, f"encoder_stage_{stage_idx + 1}"            
        elif "encoder.block." in name:
            stage_idx = int(name.split("encoder.block.")[1].split(".")[0])
            layer_id, layer_name = stage_idx + 1, f"encoder_stage_{stage_idx + 1}"         
        elif "encoder.layer_norm." in name:
            stage_idx = int(name.split("encoder.layer_norm.")[1].split(".")[0])
            layer_id, layer_name = stage_idx + 1, f"encoder_stage_{stage_idx + 1}"      
        else:
            layer_id, layer_name = 0, "other"

        if layer_id == 5:
            lr = decoder_lr
        else:
            lr = encoder_lr * (llrd_decay_rate ** (4 - layer_id))
        
        wd = 0.0 if param.ndim == 1 or name.endswith(".bias") else weight_decay
        group_key = (layer_name, lr, wd)
        if group_key not in groups:
            groups[group_key] = {"params": [], "lr": lr, "weight_decay": wd, "name": layer_name}  
        groups[group_key]["params"].append(param)
        
    return torch.optim.AdamW(list(groups.values()), lr=encoder_lr)


def get_lr_scheduler(optimiser, args, train_dl_len):
    bs, acc_steps = args.batch_size, 1
    if bs > GPU_BS:
        acc_steps = max(1, bs // GPU_BS)
        
    optimize_steps_per_epoch = train_dl_len // acc_steps
    total_optimize_steps = optimize_steps_per_epoch * args.epochs
    warmup_steps = optimize_steps_per_epoch * args.warmup_epochs
    
    kwargs = {}
    if args.lr_scheduler == "polynomial":
        kwargs["lr_end"] = 1e-8
        kwargs["power"] = 0.9

    return get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimize_steps,
        **kwargs
    )


def log_image_samples(writer, split, imgs, labels, predictions, image_size):
    cnt = min(len(imgs), 8)
    writer.log(
        {
            # f"examples/inputs/{split}": [
            #     wandb.Image(cv2.resize(
            #         np.transpose(imgs[i].detach().cpu().numpy(), axes=(1, 2, 0)), (image_size, image_size),
            #     ), caption=f"input {i}") for i in range(cnt)
            # ],
            f"examples/labels/{split}": [
                wandb.Image(labels[i].view(image_size, image_size, 1).detach().cpu().numpy(), caption=f"target {i}") for i in range(cnt)
            ],
            f"examples/predictions/{split}": [
                wandb.Image(predictions[i].view(image_size, image_size, 1).detach().cpu().numpy(), caption=f"prediction {i}") for i in range(cnt)
            ]
        }
    )


def train_one_epoch(model, ema_model, train_dl, scheduler, criterion, optimizer, args, id, curr_step=0):
    model.train()
    train_stats = {}
    optimizer.zero_grad()
    
    bs, acc_steps = args.batch_size, 1
    if bs > GPU_BS:
        acc_steps = max(1, bs // GPU_BS)

    for i, batch in enumerate(tqdm(train_dl, desc="Start training the model for one epoch...")):
        outputs = model(batch["pixel_values"].to(args.device))
        predicted_masks = outputs.logits.squeeze()
        ground_truth_masks = batch["ground_truth_mask"].float().to(args.device)
        if len(predicted_masks.shape) == 2:
            predicted_masks = predicted_masks.unsqueeze(0)
            
        loss_info = criterion(predicted_masks, ground_truth_masks)
        loss = loss_info["loss"] / acc_steps
        loss.backward()
        
        is_accumulation_step = (i + 1) % acc_steps == 0
        is_last_batch = (i + 1) == len(train_dl)
        
        if is_accumulation_step or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (args.ema):
                ema_model.update_parameters(model)
            if args.report_to == "wandb" and curr_step % 10 == 0:
                wandb.log({f"{id}_lr/{group['name']}": group["lr"] for group in optimizer.param_groups}, step=curr_step)
            curr_step += 1

        if len(train_stats) == 0:  # first batch
            train_stats = {k: v.item() for k, v in loss_info.items()}
        else:
            train_stats = {k: train_stats[k] + loss_info[k].item() for k in train_stats}

    train_stats = {k: train_stats[k] / len(train_dl) for k in train_stats}

    return train_stats["loss"], train_stats, curr_step


def eval_one_epoch(model, val_dl, criterion, epoch, writer, image_size, args, id):
    model.eval()
    val_stats = {}
    preds_gather, labels_gather = [], []

    with torch.no_grad():
        for batch_idx, val_batch in tqdm(enumerate(val_dl)):
            outputs = model(val_batch["pixel_values"].to(args.device))
            predicted_masks = outputs.logits.squeeze()

            ground_truth_masks = val_batch["ground_truth_mask"].float().to(args.device)
            if len(predicted_masks.shape) == 2:
                predicted_masks = predicted_masks.unsqueeze(0)
            loss_info = criterion(predicted_masks, ground_truth_masks)

            if len(val_stats) == 0:  # first batch
                val_stats = {k: v.item() for k, v in loss_info.items()}
            else:
                val_stats = {k: val_stats[k] + loss_info[k].item() for k in val_stats}

            if batch_idx == 0 and args.report_to == "wandb":
                predicted_masks_ = torch.nn.functional.interpolate(predicted_masks.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze()
                predicted_masks_ = torch.sigmoid(predicted_masks_)  # Convert logits to probabilities [0, 1]
                ground_truth_masks_ = torch.nn.functional.interpolate(ground_truth_masks.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze()
                log_image_samples(writer, str(id)+"_val" if epoch != -1 else str(id)+"_test", val_batch["pixel_values"], ground_truth_masks_, predicted_masks_, image_size=image_size)

            preds_gather.append(predicted_masks.detach().cpu())
            labels_gather.append(ground_truth_masks.detach().cpu())

    val_stats = {k: val_stats[k] / len(val_dl) for k in val_stats}
    preds_gather = (torch.cat(preds_gather, dim=0) > 0.0).cpu().numpy()
    labels_gather = torch.cat(labels_gather, dim=0).bool().cpu().numpy()
    seg_metrics = metrics.segmentation_metrics(preds_gather, labels_gather)

    for met in seg_metrics:
        val_stats[met] = seg_metrics[met].item()
    return val_stats["loss"], val_stats


def train_model(train_path, val_path, test_paths, writer, mod_pth, id, args):
    # config - hyperparams
    args = argparse.Namespace(**args)
    lrenc, lrdec, wd, batch_size, epochs, iou_decay_fact = args.lrenc, args.lrdec, args.wd, args.batch_size, args.epochs, args.iou_decay_fact
    start_epoch, eps_done, eps_best, curr_step = 0, epochs, 0, 0
    val_losses, val_ious, train_losses, best_val_loss, best_val_iou = [], [], [], None, None

    model_dir = os.path.join(args.save_dir, writer.id)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model, processor, image_size, mask_size = models.load_model(args.model_name, args.device)
    best_model_state = model.state_dict()
    criterion = get_criterion(args)
    optimizer = get_optimiser(model, lrenc, lrdec, wd, args.lr_layer_decay)
    batch_size = min(GPU_BS, batch_size)

    # data
    undsc = '_' if args.sub != '' else ''
    if '.csv' not in train_path:
        train_path = train_path + undsc + args.sub + '.csv'
    if '.csv' not in val_path:
        val_path = val_path + undsc + args.sub + '.csv'
        
    train_data = data.SegmentationDataset(train_path, image_size=image_size, mask_size=mask_size, transform=processor, augmentation=args.augmentation, model_name=args.model_name)
    val_data = data.SegmentationDataset(val_path, image_size=image_size, mask_size=mask_size, transform=processor, model_name=args.model_name)
    test_data_0 = data.SegmentationDataset(test_paths[0], image_size=image_size, mask_size=mask_size, transform=processor, model_name=args.model_name)
    test_data_1 = data.SegmentationDataset(test_paths[1], image_size=image_size, mask_size=mask_size, transform=processor, model_name=args.model_name)
    test_data_2 = data.SegmentationDataset(test_paths[2], image_size=image_size, mask_size=mask_size, transform=processor, model_name=args.model_name)
    
    train_dl = DataLoader(train_data, batch_size=batch_size, num_workers=args.workers, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, num_workers=args.workers, shuffle=False)
    test_dl_0 = DataLoader(test_data_0, batch_size=batch_size, num_workers=args.workers, shuffle=False)
    test_dl_1 = DataLoader(test_data_1, batch_size=batch_size, num_workers=args.workers, shuffle=False)
    test_dl_2 = DataLoader(test_data_2, batch_size=batch_size, num_workers=args.workers, shuffle=False)

    scheduler = get_lr_scheduler(optimizer, args, len(train_dl))
    ema_model = model
    if (args.ema):
        ema_avg_fn = get_ema_multi_avg_fn()
        ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)

    # load model
    if mod_pth:
        resume_ckpt = torch.load(mod_pth)
        model.load_state_dict(resume_ckpt)
        ema_model = model
        if (args.ema):
            ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)
        print('loaded ready model', mod_pth)

    # training for epochs
    for i in range(start_epoch, epochs):
        train_loss, train_stats, curr_step = train_one_epoch(model, ema_model, train_dl, scheduler, criterion, optimizer, args, id, curr_step)
        val_loss_base, val_stats_base = eval_one_epoch(model, val_dl, criterion, i, writer, image_size, args, f"{id}_base")
        val_loss_ema, val_stats_ema = val_loss_base, val_stats_base
        if (args.ema):
            val_loss_ema, val_stats_ema = eval_one_epoch(ema_model, val_dl, criterion, i, writer, image_size, args, f"{id}_ema")
        val_iou = val_stats_ema["iou"]
        val_loss = val_stats_ema["loss"]

        print(f"Epoch: {i}, train loss: {train_loss:.4f}, base val_loss: {val_loss_base:.4f}, ema val_loss: {val_loss_ema:.4f}, ema val_IoU: {val_iou:.4f}")        
        print(f'ph_{id}', f'ep_{i}', 'train', train_stats)
        print(f'ph_{id}', f'ep_{i}', 'val', val_stats_base)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        if args.report_to == "wandb":
            log_stats = {f'{id}_train/{k}': v for k, v in train_stats.items()}
            log_stats.update({f'{id}_val_base/{k}': v for k, v in val_stats_base.items()})
            log_stats.update({f'{id}_val_ema/{k}': v for k, v in val_stats_ema.items()})
            log_stats[f'{id}_epoch'] = i
            wandb.log(log_stats)

        if best_val_iou is not None and val_iou < iou_decay_fact * best_val_iou:
            print('iou decay threshold hit!', val_iou, best_val_loss, 'ep', i)
            eps_done = i + 1
            break
        if best_val_loss is None or val_iou > best_val_iou:
            print('outperformed!', best_val_iou, val_iou)
            eps_best = i
            best_val_loss = val_loss
            best_val_iou = val_iou
            clean_state_dict = model.state_dict()
            if (args.ema):
                raw_ema_state = ema_model.state_dict()
                clean_state_dict = {}
                for key, value in raw_ema_state.items():
                    if key == "n_averaged":
                        continue
                    clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key
                    clean_state_dict[clean_key] = value
            best_model_state = clean_state_dict

    # save tr-val results and the best model
    mod_pthn = os.path.join(model_dir, f'model_ph{id}.pth')
    txt_pth = os.path.join(model_dir, f'stats_ph{id}.txt')
    torch.save(best_model_state, mod_pthn)
    model.load_state_dict(best_model_state)
    print('phase stats start, name, loss, iou best')
    print(mod_pthn, best_val_loss, best_val_iou)
    print('val_losses', val_losses)
    print('val_ious', val_ious)
    print('train_losses', train_losses)
    print('phase stats end')

    # fin test
    test_lossS, test_statsS = eval_one_epoch(model, test_dl_0, criterion, -1, writer, image_size, args, id)
    print(f"synt-Epoch: {epochs}, test_loss: {test_lossS}" + ", test_dice: {}, test_IoU: {}".format(test_statsS["dice"], test_statsS["iou"]))
    test_lossR, test_statsR = eval_one_epoch(model, test_dl_1, criterion, -1, writer, image_size, args, id)
    print(f"rzecz-Epoch: {epochs}, test_loss: {test_lossR}" + ", test_dice: {}, test_IoU: {}".format(test_statsR["dice"], test_statsR["iou"]))
    test_lossG, test_statsG = eval_one_epoch(model, test_dl_2, criterion, -1, writer, image_size, args, id)
    print(f"GDA-Epoch: {epochs}, test_loss: {test_lossG}" + ", test_dice: {}, test_IoU: {}".format(test_statsG["dice"], test_statsG["iou"]))

    if args.report_to == "wandb":
      wandb.log({f'{id}_test/SYNT/{k}': v for k, v in test_statsS.items()})
      wandb.log({f'{id}_test/DK/{k}': v for k, v in test_statsR.items()})
      wandb.log({f'{id}_test/GDA/{k}': v for k, v in test_statsG.items()})

    # txt results
    with open(txt_pth, 'w') as f:
        f.write(f'val_losses {str(val_losses)}\n')
        f.write(f'val_ious {str(val_ious)}\n')
        f.write(f'train_losses {str(train_losses)}\n')
        f.write(f'best_val_loss {best_val_loss}\n')
        f.write(f'best_val_iou {best_val_iou}\n')
        f.write(f'eps_best {eps_best}\n')
        f.write(f'eps_done {eps_done}\n')
        iou = test_statsS["iou"]
        f.write(f"synt-Epoch: {epochs}, test_loss: {test_lossS}, test_IoU: {iou}\n")
        iou = test_statsR["iou"]
        f.write(f"rzecz-Epoch: {epochs}, test_loss: {test_lossR}, test_IoU: {iou}\n")
        iou = test_statsG["iou"]
        f.write(f"GDA-Epoch: {epochs}, test_loss: {test_lossG}, test_IoU: {iou}\n")

    return mod_pthn
