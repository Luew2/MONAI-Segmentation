#! /usr/bin/python

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandAffined,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    ToTensord,
)
from monai.metrics import compute_meandice
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import sys
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

from common import *


def run(param, train_files, val_files):
    
    #--------------------------------------------------------------------------------
    # Prepare tensorboard
    #--------------------------------------------------------------------------------
    
    # Tensorboard
    if param.use_tensorboard == 1:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('runs/segmentation_experiment_1')
    
        
    torch.multiprocessing.set_sharing_strategy('file_system')
    print_config()
    
    set_determinism(seed=0)
    
    
    #--------------------------------------------------------------------------------
    # Train/validation datasets
    #--------------------------------------------------------------------------------
    
    val_transforms = loadValidationTransforms(param)
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=param.pixel_dim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="LPS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=param.pixel_intensity_min, a_max=param.pixel_intensity_max,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            # ScaleIntensityRangePercentilesd(
            #     keys=["image"], lower=param.pixel_intensity_percentile_min, upper=param.pixel_intensity_percentile_max,
            #     b_min=0.0, b_max=1.0, clip=True,
            # ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                #spatial_size=(96,96,96),
                #spatial_size=(32, 32, 16),
                spatial_size=param.window_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            #RandAffined(
            #    keys=['image', 'label'],
            #    mode=('bilinear', 'nearest'),
            #    prob=1.0,
            #    #spatial_size=(96, 96, 96),
            #    spatial_size=(64, 64, 16),
            #    rotate_range=(0, 0, np.pi/15),
            #    scale_range=(0.1, 0.1, 0.1)),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    
    
    #--------------------------------------------------------------------------------
    # Training
    #--------------------------------------------------------------------------------
    
    (model_unet, post_pred, post_label) = setupModel()
    
    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device(param.training_device_name)
    model = model_unet.to(device)
    
    # Loss function & optimizer
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    
    for epoch in range(param.max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{param.max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    #roi_size = (160, 160, 160)
                    roi_size = param.window_size
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        param.root_dir, param.model_file))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                    )
    
                if param.use_tensorboard == 1:
                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                    writer.add_scalar("Mean Dice", metric, epoch)
    
                    # write to tensorboard
                    #img_grid = torchvision.utils.make_grid(val_labels)
                    #writer.add_image('segmentation', img_grid)
                    writer.flush()
    
    print(f"train completed, best_metric: {best_metric:.4f} "
          f"at epoch: {best_metric_epoch}")
    
    if param.use_matplotlib == 1:
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [val_interval * (i + 1) for i in range(len(metric_values))]
        y = metric_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.show()


def main(argv):
    
  try:
    parser = argparse.ArgumentParser(description="Apply a saved DL model for segmentation.")
    parser.add_argument('cfg', metavar='CONFIG_FILE', type=str, nargs=1,
                        help='Configuration file')
    #parser.add_argument('input', metavar='INPUT_PATH', type=str, nargs=1,
    #help='A file or a folder that contains images.')
            
    args = parser.parse_args(argv)

    config_file = args.cfg[0]
    #input_path = args.input[0]

    print('Loading parameters from: ' + config_file)
    param = TrainingParam(config_file)

    train_files = generateLabeledFileList(param, 'train')
    val_files = generateLabeledFileList(param, 'val')
    
    n_train = len(train_files)
    n_val = len(val_files)
    print('Training data size: ' + str(n_train))
    print('Validation data size: ' + str(n_val))    

    run(param, train_files, val_files)


  except Exception as e:
    print(e)
  sys.exit()


if __name__ == "__main__":
  main(sys.argv[1:])
    
