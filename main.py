# inspired from https://github.com/thuml/CDAN , https://github.com/ZJULearning/ALDA and https://github.com/kilianFatras/JUMBOT/tree/main/Domain_Adaptation
import json
import random
import torch
import torch.optim as optim
import wandb
import numpy as np
import os

# from model.unet import UNet
from train import Train
from data_loader.dataset_ot import Dataset
import gc

# reading config file
with open(
    "/share/projects/erasmus/pratichhya_sharma/version00/utils/config.json",
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
# net = UNet(config["n_channel"], config["n_classes"])
# net.load_state_dict(torch.load(config["model_path"] + "noDA_wien.pt"))

from seg_model_smp.models_predefined import segmentation_models_pytorch as psmp
net = psmp.UnetPlusPlus(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)
net.cuda()

saving_interval = 10
NUM_EPOCHS = config["epoch"]
lrs = []


def main(
    net,
):
    set_seed(40)

    # seting training and testing dataset
    dsource_loaders = Dataset(config["data_folder"], config["patchsize"], "both")
    dsource_loaders.array_torch()
    source_dataloader = dsource_loaders.source_dataloader
    val_source_dataloader = dsource_loaders.valid_source_dataloader

    #dtarget_loaders = Dataset(config["data_folder"], config["patchsize"], "training_target")
    #dtarget_loaders.array_torch()
    target_dataloader = dsource_loaders.target_dataloader
    val_target_dataloader = dsource_loaders.valid_target_dataloader

    # computing the length
    len_train_source = len(source_dataloader)  # training steps
    len_train_target = len(target_dataloader)
    print(
        f"length of train source:{len_train_source}, lenth of train target is {len_train_target}"
    )
    # computing the length
    len_val_source = len(val_source_dataloader)  # training steps
    len_val_target = len(val_target_dataloader)
    print(
        f"length of validation source:{len_val_source}, lenth of validation target is {len_val_target}"
    )

    parameter_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"The model has {parameter_num:,} trainable parameters")

    ## set optimizer
    # optimizer = optim.SGD(
    #     net.parameters(), lr=config["base_lr"],momentum=0.66, weight_decay=0.0005
    # )
    optimizer=optim.Adam(net.parameters(),lr=config["base_lr"])

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    # We define the scheduler
    schedule_param = config["lr_param"]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [300, 1000, 20000], gamma=schedule_param["gamma"]
    )
    
    patience = 7
    the_last_loss = 100
    trigger_times = 0
    for e in range(1, NUM_EPOCHS + 1):
        print("----------------------Traning phase-----------------------------")
        train_loss, acc_mat = Train.train_epoch(net,optimizer, source_dataloader, target_dataloader)
        print(f"Training loss in average for epoch {str(e)} is {train_loss}")
        print(f"Training F1 in average for epoch {str(e)} is {acc_mat[0]}")
        print(f"Training Accuracy in average for epoch {str(e)} is {acc_mat[1]}")
        print(f"Training IOU in average for epoch {str(e)} is {acc_mat[2]}")
        print(f"Training K in average for epoch {str(e)} is {acc_mat[3]}")
        # wandb.log({'Train Loss': train_loss,'Train_F1': acc_mat[0],'Train_acc':acc_mat[1],'Train_IoU':acc_mat[2],'Train_Kappa':acc_mat[3]}, step = e)
        # (total/batch)*epoch=iteration
        del train_loss,acc_mat
        print("----------------------Evaluation phase-----------------------------")
        valid_loss, acc_mat = Train.eval_epoch(e, net, source_dataloader, target_dataloader)
        print(f"Evaluation loss in average for epoch {str(e)} is {valid_loss}")
        print(f"Evaluation F1 in average for epoch {str(e)} is {acc_mat[0]}")
        print(f"Evaluation Accuracy in average for epoch {str(e)} is {acc_mat[1]}")
        print(f"Evaluation IOU in average for epoch {str(e)} is {acc_mat[2]}")
        print(f"Evaluation K in average for epoch {str(e)} is {acc_mat[3]}")
        # wandb.log({'Val_Loss': valid_loss,'Val_F1': acc_mat[0],'Val_acc':acc_mat[1],'Val_IoU':acc_mat[2],'Val_Kappa':acc_mat[3]}, step = e)

        # Decay Learning Rate kanxi: check this
        if e % 10 == 0:
            scheduler.step()
        # Print Learning Rate
        print("last learning rate:", scheduler.get_last_lr(), "LR:", scheduler.get_lr())

        # Early stopping
        print("###################### Early stopping ##########################")
        the_current_loss = valid_loss
        print("The current validation loss:", the_current_loss)
        
        if the_current_loss >= the_last_loss:
            trigger_times += 1
            torch.save(net.state_dict(), config["model_path"] + "es_trig1_weinv3.pt")
            print("trigger times:", trigger_times)
            if trigger_times == patience:
                print("Early stopping!\nStart to test process.")
                torch.save(net.state_dict(), config["model_path"] + "es_wienv3.pt")
                break
        else:
            print(f"trigger times: {trigger_times}")
            the_last_loss = the_current_loss
            
#         del valid_loss, acc_mat
#         # lrs.append(optimizer.param_groups[0]["lr"])
        # print("learning rates are:",lrs
    print("finished")
    torch.save(net.state_dict(), config["model_path"] + "DA_wienv3.pt")
    gc.collect()
    torch.cuda.empty_cache()



if __name__ == "__main__":
    torch.cuda.empty_cache()
    wandb.login()
    wandb.init(project="server")
    main(net)