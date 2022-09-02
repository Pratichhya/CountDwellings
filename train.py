import random
import torch
import torch.optim as optim
import wandb
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile as tiff

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torch.nn import MaxPool1d

# files
from data_loader.dataset import Dataset


def rmsefn(y, yhat):
    return torch.sqrt(torch.mean((yhat-y)**2))


lossfn = torch.nn.MSELoss()

# early stopping patience; how long to wait after last time validation loss improved.
patience = 5
the_last_loss = 100


def train_epoch(net, optimizer, dataloader):
    len_train = len(dataloader)
    total_loss = 0
    net.train()
    iter_ = 0

    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len_train):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        # zero optimizer
        optimizer.zero_grad()
        output = net(data)

        loss = lossfn(output, target)
        loss.backward()
        optimizer.step()

        # add evaluation metric here
        total_loss += loss
        # print(output.data.cpu().numpy().shape)
        no_of_buildings = np.rint(
            np.sum(output.data.cpu().numpy(), axis=(1, 2, 3)))
        actual_number = np.sum(target.data.cpu().numpy(), axis=(1, 2, 3))
        rmse = rmsefn(target, output)
        # wandb.log({'train_Loss': loss,'train_F1': f1_source_step,'train_acc':acc_step,'train_IoU':IoU_step})
    return (total_loss/len_train), rmse, no_of_buildings, actual_number


def eval_epoch(net, epochs, dataloader):
    len_train = len(dataloader)
    f1_source, acc, IoU, K = 0.0, 0.0, 0.0, 0.0
    val_loss = 0
    net.eval()
    iter_ = 0
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len_train):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output = net(data)
            loss = lossfn(output, target)
            # add evaluation metric here
            rmse = rmsefn(target, output)
            total_loss += loss
            no_of_buildings = np.rint(
                np.sum(output.data.cpu().numpy(), axis=(1, 2, 3)))
            actual_number = np.sum(target.data.cpu().numpy(), axis=(1, 2, 3))
            # density_map = output.data.cpu().numpy()[0]
            # plt.imshow(np.swapaxes(density_map,0,2))
            # plt.show()

    return (total_loss/len_train), rmse, no_of_buildings, actual_number
