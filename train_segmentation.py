
from seg_model_smp.models_predefined import segmentation_models_pytorch as psmp
import json
import random
import torch
import torch.optim as optim
import wandb
import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
import tifffile as tiff
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable

# files
from data_loader.dataset import Dataset
from utils.criterion import DiceLoss
from utils import eval_metrics
# from model.unet import UNet


Dice = DiceLoss()
# reading config file
with open(
    "/share/projects/erasmus/pratichhya_sharma/DAoptim/DAoptim/utils/config.json",
    "r",
) as read_file:
    config = json.load(read_file)


def set_seed(seed):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


# set network
net = psmp.Unet(encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=3,
                # model output channels (number of classes in your dataset)
                classes=1,
                )
# net = UNet(config["n_channel"], config["n_classes"])
net.cuda()

saving_interval = 10
base_lr = 0.1
running_loss = 0.0
testing_loss = 0.0
training_loss = 0.0
validation_loss = 0.0
mode = config["mode"]
NUM_EPOCHS = 50
lrs = []

# early stopping patience; how long to wait after last time validation loss improved.
patience = 5
the_last_loss = 100


def train_epoch(optimizer, dataloader):
    len_train = len(dataloader)
    f1_source, acc, IoU, K = 0.0, 0.0, 0.0, 0.0
    total_loss = 0
    net.train()
    iter_ = 0

    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len_train):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        # zero optimizer
        optimizer.zero_grad()
        _, output = net(data)
        del data
        loss = Dice(output, target)
        loss.backward()
        optimizer.step()
        # evaluation
        f1_source_step, acc_step, IoU_step, K_step = eval_metrics.f1_score(
            target, output)
        f1_source += f1_source_step
        acc += acc_step
        IoU += IoU_step
        K += K_step
        total_loss += loss
        del loss, K_step, IoU_step, acc_step, f1_source_step
        # wandb.log({'train_Loss': loss,'train_F1': f1_source_step,'train_acc':acc_step,'train_IoU':IoU_step})
    return (total_loss/len_train), [f1_source/len_train, acc/len_train, IoU/len_train, K/len_train]


def eval_epoch(epochs, dataloader):
    len_train = len(dataloader)
    f1_source, acc, IoU, K = 0.0, 0.0, 0.0, 0.0
    val_loss = 0
    net.eval()
    iter_ = 0
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len_train):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            _, output = net(data)
            loss = Dice(output, target)

            # evaluation
            f1_source_step, acc_step, IoU_step, K_step = eval_metrics.f1_score(
                target, output)
            f1_source += f1_source_step
            acc += acc_step
            IoU += IoU_step
            K += K_step
            total_loss += loss
#             if iter_ % 100 == 0:
            # if epochs % 5 == 0:
# #                 clear_output()
#                 # rgb = data.data.cpu().numpy()[0]
#                 pred = output.data.cpu().numpy()[0]
#                 gt = target.data.cpu().numpy()[0]
#                 visualize_predict(np.moveaxis(rgb,0,2),np.moveaxis(gt,0,2), np.moveaxis(pred,0,2))
#                 plt.show()
            # images_pred = wandb.Image(pred, caption="Top: Output, Bottom: Input")
            # # images_rgb = wandb.Image(rgb, caption="Top: Output, Bottom: Input")
            # images_gt = wandb.Image(gt, caption="Top: Output, Bottom: Input")
            # wandb.log({"Ground truth": images_gt,"Prediction": images_pred})

#             iter_ += 1
            # wandb.log({'Val_Loss': loss,'Val_F1': f1_source_step,'Val_acc':acc_step,'Val_IoU':IoU_step})
        del loss, K_step, IoU_step, acc_step, f1_source_step
    return (total_loss/len_train), [f1_source/len_train, acc/len_train, IoU/len_train, K/len_train]


def SimpleUnet(net):
    set_seed(0)
    parameter_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"The model has {parameter_num:,} trainable parameters")

    # set optimizer
    optimizer = optim.SGD(
        net.parameters(), lr=0.1, momentum=0.66, weight_decay=0.005
    )
    # optimizer=optim.Adam(net.parameters(),lr=0.01)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    # We define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [1, 10, 20], gamma=0.01)

    patience = 3
    the_last_loss = 50
    test_f1 = 0
    trigger_times = 0
    # seting training and testing dataset
    dsource_loaders = Dataset(
        config["data_folder"], config["patchsize"], mode["1"])
    dsource_loaders.array_torch()
    source_dataloader = dsource_loaders.source_dataloader
    val_source_dataloader = dsource_loaders.valid_source_dataloader

    for e in range(1, NUM_EPOCHS + 1):
        print("----------------------Traning phase-----------------------------")
        train_loss, t_acc_mat = train_epoch(optimizer, source_dataloader)
        print(f"Training loss in average for epoch {str(e)} is {train_loss}")
        print(f"Training F1 in average for epoch {str(e)} is {t_acc_mat[0]}")
        print(
            f"Training Accuracy in average for epoch {str(e)} is {t_acc_mat[1]}")
        print(f"Training IOU in average for epoch {str(e)} is {t_acc_mat[2]}")
        # print(f"Training K in average for epoch {str(e)} is {acc_mat[3]}")
        # wandb.log({'Train Loss': train_loss,'Train_F1': acc_mat[0],'Train_acc':acc_mat[1],'Train_IoU':acc_mat[2]})
        # (total/batch)*epoch=iteration

        print("----------------------Evaluation phase-----------------------------")
        valid_loss, acc_mat = eval_epoch(e, val_source_dataloader)
        print(f"Evaluation loss in average for epoch {str(e)} is {valid_loss}")
        print(f"Evaluation F1 in average for epoch {str(e)} is {acc_mat[0]}")
        print(
            f"Evaluation Accuracy in average for epoch {str(e)} is {acc_mat[1]}")
        print(f"Evaluation IOU in average for epoch {str(e)} is {acc_mat[2]}")
        # print(f"Evaluation K in average for epoch {str(e)} is {acc_mat[3]}")
        wandb.log({'Val_Loss': valid_loss, 'Val_F1': acc_mat[0], 'Val_acc': acc_mat[1], 'Val_IoU': t_acc_mat[2],
                   'Train Loss': train_loss, 'Train_F1': t_acc_mat[0], 'Train_acc': t_acc_mat[1], 'Train_IoU': t_acc_mat[2]})
        # Decay Learning Rate kanxi: check this
        # if e % 10 == 0:
        #     scheduler.step()
        # Print Learning Rate
        # print("last learning rate:", scheduler.get_last_lr(), "LR:", scheduler.get_lr())

        # Early stopping
        print("###################### Early stopping ##########################")
        the_current_loss = valid_loss
        print("The current validation loss:", the_current_loss)

        if the_current_loss >= the_last_loss:
            trigger_times += 1
            if test_f1 <= acc_mat[0]:
                test_f1 = acc_mat[0]
                torch.save(net.state_dict(),
                           config["model_path"] + "f1_simple_minawao.pt")
            print("trigger times:", trigger_times)
            if trigger_times == patience:
                print("Early stopping!\nStart to test process.")
                torch.save(net.state_dict(),
                           config["model_path"] + "es_simple_minawao.pt")
        else:
            print(f"trigger times: {trigger_times}")
            the_last_loss = the_current_loss

        del valid_loss, acc_mat, train_loss, t_acc_mat
        # lrs.append(optimizer.param_groups[0]["lr"])
        # print("learning rates are:",lrs
    torch.save(net.state_dict(), config["model_path"] + "simple_minawao.pt")
    print("finished")


if __name__ == "__main__":
    wandb.login()
    wandb.init(project="simple_final")
    SimpleUnet(net)
