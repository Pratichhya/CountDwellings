# inspired from https://github.com/NeuroSYS-pl/objects_counting_dmap/tree/7d91bb865fe00f1c8eca17bcdac693162a981c77
import json
import random
import torch
import torch.optim as optim
import wandb
import numpy as np
import os
import time

import train
from model import UNet
from data_loader.dataset import Dataset
import gc


def set_seed(seed):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    return True


# set network
net = UNet(input_filters=3)
net.cuda()


saving_interval = 10
NUM_EPOCHS = 90
base_lr = 0.001


def main(net):
    set_seed(42)
    parameter_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"The model has {parameter_num:,} trainable parameters")

    # set optimizer
    optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0005)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    # We define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [1, 10, 20], gamma=0.1)

    # calling the dataloader
    d_loaders = Dataset("./data_loader")
    d_loaders.array_torch()
    train_data = d_loaders.train_data
    val_data = d_loaders.val_data

    for e in range(1, NUM_EPOCHS + 1):
        print(
            f"----------------------Traning phase {e} -----------------------------")
        train_loss, rmse, no_of_buildings, actual_number = train.train_epoch(
            net, optimizer, train_data)
        print(f"Training Loss in {e} is {train_loss}")
        print(f"Root mean squared error is {rmse}")
        print(
            f"Total number of buildings during training is {no_of_buildings}/{actual_number}")
        # wandb.log({'Train loss': train_loss,'Train RMSE': rmse,'Building Count':no_of_buildings}, step = e)
        del train_loss, rmse, no_of_buildings, actual_number
        print(
            f"----------------------Evaluation phase {e} -----------------------------")
        valid_loss, rmse, no_of_buildings, actual_number = train.eval_epoch(
            net, e, val_data)
        print(f"Evaluation Loss in {e} is {valid_loss}")
        print(f"Root mean squared error is {rmse}")
        sum_predicted = sum(no_of_buildings)
        sum_actual = sum(actual_number)
        with open("./count_record.txt", "a") as a_file:
            a_file.write("\n")
            a_file.write(
                f"Total number of buildings observed is {sum_predicted}/{sum_actual} during epoch {e} Evaluation phase")
        wandb.log({'Valid loss': valid_loss, 'Val RMSE': rmse}, step=e)
        del valid_loss, rmse, no_of_buildings, actual_number

    torch.save(net.state_dict(), "saved_model/countmodel_final.pt")
    print("finished")


if __name__ == "__main__":
    wandb.login()
    wandb.init(project="CD")
    main(net)
