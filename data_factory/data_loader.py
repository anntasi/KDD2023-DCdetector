import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train": #train and val did not use label
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")[:,:]
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")[:,:]
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")[:]

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

        
        
class UCRSegLoader(object):
    def __init__(self, index, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.index = index
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/UCR_"+str(index)+"_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/UCR_"+str(index)+"_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/UCR_"+str(index)+"_test_label.npy")
        if self.mode == "val":
            print("train:", self.train.shape)
            print("test:", self.test.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])    
        

class UCRAUGSegLoader(object):
    def __init__(self, index, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.index = index
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/UCR_AUG_"+str(index)+"_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/UCR_AUG_"+str(index)+"_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/UCR_AUG_"+str(index)+"_test_label.npy")
        if self.mode == "val":
            print("train:", self.train.shape)
            print("test:", self.test.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]) 
        

class NIPS_TS_WaterSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Water_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])        
        
        
        
class NIPS_TS_SwanSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]) 
        

class NIPS_TS_CCardSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_CCard_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_CCard_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_CCard_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]) 
        

        
        
class SMD_OriSegLoader(object):
    def __init__(self, index, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.index = index
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_Ori_"+str(index)+"_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_Ori_"+str(index)+"_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMD_Ori_"+str(index)+"_test_label.npy")
        if self.mode == "val":
            print("train:", self.train.shape)
            print("test:", self.test.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])         

class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class TNSSegLoaderV2(Dataset):
    """
    Loader for fixed dataset layout:

    expdata/
        train/
            session_train.npy
        test_normal/
            session_test_normal.npy
        test_anomaly/
            session_test_anomaly.npy

    Each .npy file is one session, shape = (T, C).
    """

    def __init__(self, root_path, win_size, step=1, mode="train", drop_col_idx=None):
        self.root_path = root_path
        self.win_size = int(win_size)
        self.step = int(step)
        self.mode = str(mode).lower()
        self.drop_col_idx = drop_col_idx

        valid_modes = {"train", "val", "test", "thre"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported mode={self.mode}, expected one of {valid_modes}")

        # -----------------------------
        # 1. 決定目前 mode 要讀哪些檔案
        # -----------------------------
        if self.mode == "train":
            session_files = [
                os.path.join(root_path, "train", "session_train.npy")
            ]
            session_labels = [0]

        elif self.mode == "val":
            session_files = [
                os.path.join(root_path, "test_normal", "session_test_normal.npy")
            ]
            session_labels = [0]

        elif self.mode == "thre":
            session_files = [
                os.path.join(root_path, "test_normal", "session_test_normal.npy")
            ]
            session_labels = [0]

        elif self.mode == "test":
            session_files = [
                os.path.join(root_path, "test_normal", "session_test_normal.npy"),
                os.path.join(root_path, "test_anomaly", "session_test_anomaly.npy"),
            ]
            session_labels = [0, 1]

        # 檢查檔案是否存在
        for fp in session_files:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Missing session file: {fp}")

        # -----------------------------
        # 2. 用 train session fit scaler
        # -----------------------------
        train_fp = os.path.join(root_path, "train", "session_train.npy")
        if not os.path.exists(train_fp):
            raise FileNotFoundError(f"Missing training session file: {train_fp}")

        train_arr = np.load(train_fp).astype(np.float32)

        if train_arr.ndim != 2:
            raise ValueError(f"{train_fp} must be 2D, got shape={train_arr.shape}")

        if self.drop_col_idx is not None:
            train_arr = np.delete(train_arr, int(self.drop_col_idx), axis=1)

        train_arr = np.nan_to_num(train_arr, nan=0.0, posinf=0.0, neginf=0.0)

        self.scaler = StandardScaler()
        self.scaler.fit(train_arr)

        # -----------------------------
        # 3. 載入當前 mode 的 session，並切 windows
        # -----------------------------
        self.windows = []
        self.window_labels = []#widow內的packet的label
        self.window_packet_indices = []#每個 window 對應的 packet index

        # packet-level evaluation 會用到
        self.packet_labels_raw = []#packet 的 ground truth label
        self.selected_packet_mask = []

        global_packet_idx = 0

        for session_fp, session_label in zip(session_files, session_labels):
           
            arr = np.load(session_fp).astype(np.float32)
            session_window_count = 0
            if arr.ndim != 2:
                raise ValueError(f"{session_fp} must be 2D, got shape={arr.shape}")

            if self.drop_col_idx is not None:
                arr = np.delete(arr, int(self.drop_col_idx), axis=1)

            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = self.scaler.transform(arr).astype(np.float32)

            T, D = arr.shape

            # 整個 session 的 packet label 都一樣
            pkt_labels = np.full((T,), session_label, dtype=np.int64)

            # 先收集 packet-level GT
            self.packet_labels_raw.extend(pkt_labels.tolist())

            # 這個 session 裡哪些 packet 有被 window 覆蓋到
            local_mask = np.zeros((T,), dtype=bool)

            if T >= self.win_size:
                for start in range(0, T - self.win_size + 1, self.step):
                    session_window_count += 1

                    x = arr[start:start + self.win_size]          # shape = (win_size, D)
                    y = pkt_labels[start:start + self.win_size]   # shape = (win_size,)

                    pkt_idx = np.arange(
                        global_packet_idx + start,
                        global_packet_idx + start + self.win_size,
                        dtype=np.int64
                    )

                    self.windows.append(x)
                    self.window_labels.append(y.astype(np.float32))
                    self.window_packet_indices.append(pkt_idx)

                    local_mask[start:start + self.win_size] = True
            print(
            f"[DEBUG][{self.mode}] session={os.path.basename(session_fp)} "
            f"T={T} label={session_label} windows_added={session_window_count} "
            f"local_mask_sum={local_mask.sum()} "
            f"global_idx_range=({global_packet_idx},{global_packet_idx + T - 1})",
            flush=True
        )

            if session_window_count > 0:
                print(
                    f"[DEBUG][{self.mode}] last window pkt_idx = {pkt_idx}",
                    flush=True
                )
            self.selected_packet_mask.extend(local_mask.tolist())
            global_packet_idx += T

        # -----------------------------
        # 4. 轉成 numpy array
        # -----------------------------
        if len(self.windows) == 0:
            self.windows = np.empty((0, self.win_size, train_arr.shape[1]), dtype=np.float32)
            self.window_labels = np.empty((0, self.win_size), dtype=np.float32)
            self.window_packet_indices = np.empty((0, self.win_size), dtype=np.int64)
        else:
            self.windows = np.asarray(self.windows, dtype=np.float32)
            self.window_labels = np.asarray(self.window_labels, dtype=np.float32)
            self.window_packet_indices = np.asarray(self.window_packet_indices, dtype=np.int64)

        self.packet_labels_raw = np.asarray(self.packet_labels_raw, dtype=np.int64)
        self.selected_packet_mask = np.asarray(self.selected_packet_mask, dtype=bool)

        print(f"[TNS_V2:{self.mode}] num_windows = {len(self.windows)}")
        print(f"[TNS_V2:{self.mode}] total_packets = {len(self.packet_labels_raw)}")

        if len(self.windows) > 0:
            print(f"[TNS_V2:{self.mode}] first window shape = {self.windows[0].shape}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = self.window_labels[idx]
        pkt_idx = self.window_packet_indices[idx]

        return (
            x.astype(np.float32),
            y.astype(np.float32),
            pkt_idx.astype(np.int64)
        )
    
def get_loader_segment(index, data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset =='SWAT'):
        dataset = SWATSegLoader(data_path,win_size,1,mode)
    elif (dataset == 'UCR'):
        dataset = UCRSegLoader(index, data_path, win_size, 1, mode)
    elif (dataset == 'UCR_AUG'):
        dataset = UCRAUGSegLoader(index, data_path, win_size, 1, mode)
    elif (dataset == 'NIPS_TS_Water'):
        dataset = NIPS_TS_WaterSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'NIPS_TS_Swan'):
        dataset = NIPS_TS_SwanSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'NIPS_TS_CCard'):
        dataset = NIPS_TS_CCardSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMD_Ori'):
        dataset = SMD_OriSegLoader(index, data_path, win_size, 1, mode)
    elif dataset == 'TNS':
        dataset = TNSSegLoaderV2(
        root_path=data_path,
        win_size=win_size,
        step=1,
        mode=mode,
        drop_col_idx=None
    )
    shuffle = False
    if mode == 'train':
        shuffle = True

    if mode == 'train':
        drop_last = True
    else:
        drop_last = False
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8,
                             drop_last=drop_last)
    return data_loader
