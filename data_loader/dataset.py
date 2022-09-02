# importing necessary packages

import os
import rasterio
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import shutil
import json

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader, random_split


class Dataset():
    def __init__(self, data_folder):
        # connecting to the folder
        print("Buckle up, here with start the journey🚲")
        self.data_folder = data_folder

    def array_torch(self):
        print("On the way to find data and make our final dataloader")
        self.Xmain = np.load(self.data_folder + "/used_data/Xdata_128CD.npy")
        self.Ymain = np.load(self.data_folder + "/used_data/Ydata_128CD.npy")
        # self.Xmain = self.Xmain[:80,:,:,:]
        # self.Ymain = self.Ymain[:80,:,:,:]
        print("----------------------Found already existing npy----------------------")

        print("shape of Xmain: ", self.Xmain.shape)
        print("shape of Ymain: ", self.Ymain.shape)
#         print(f"x max:{self.Xmain.max()}")
#         print(f"x min:{self.Xmain.min()}")
#         print(f"y max:{self.Ymain.max()}")
#         print(f"y min:{self.Ymain.min()}")

        print("----------------------------------------------------------------------")

        # set aside 25% of train and val data for evaluation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.Xmain, self.Ymain, test_size=0.25, random_state=8)

        # printing final shape
        print(f"shape of Training Images{self.X_train.shape}")
        print(f"shape of Training Labels{self.y_train.shape}")
        print(f"shape of Validation Images{self.X_val.shape}")
        print(f"shape of Validation Labels{self.y_val.shape}")

        # converting numpy array to pytorch dataset
        self.x_train = torch.Tensor(self.X_train.astype(np.float16))
        self.y_train = torch.Tensor(self.y_train.astype(np.float16))
        self.x_val = torch.Tensor(self.X_val.astype(np.float16))
        self.y_val = torch.Tensor(self.y_val.astype(np.float16))

        self.tensor_train = TensorDataset(self.x_train, self.y_train)
        self.train_data = DataLoader(self.tensor_train, batch_size=8,
                                     pin_memory=True, shuffle=True, worker_init_fn=np.random.seed(42))
        self.tensor_val = TensorDataset(self.x_val, self.y_val)
        self.val_data = DataLoader(self.tensor_val, batch_size=8,
                                   pin_memory=True, shuffle=True, worker_init_fn=np.random.seed(42))

        print("Finally atleast test dataloader section works 😌")


# if __name__ == "__main__":
#     DATASET = Dataset("/share/projects/erasmus/pratichhya_sharma/CD/data_loader")
#     DATASET.array_torch()
