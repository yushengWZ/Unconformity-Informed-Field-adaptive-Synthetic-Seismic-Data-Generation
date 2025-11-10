import torch
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import random
import cv2



def histogram_equalization(data):
    H, W, D = data.shape
    equalized_data = np.zeros_like(data, dtype=np.float32)
    for d in range(D):
        img = data[:, :, d]
        hist, bins = np.histogram(img.ravel(), bins=256, range=[0, img.max()])
        cdf = hist.cumsum() 
        cdf = cdf / cdf[-1]  
        equalized_img = np.interp(img.ravel(), bins[:-1], cdf * (img.max() - img.min()) + img.min())
        equalized_img = equalized_img.reshape(img.shape)
        equalized_img = (equalized_img - equalized_img.min()) / (equalized_img.max() - equalized_img.min())
        equalized_data[:, :, d] = equalized_img
    return equalized_data




def his_dataset():
    # source_labels
    file_str = r'data_for_training\unconf_cut.npy'
    source_data = np.load(file_str).astype(np.float32)
    source_data = histogram_equalization(source_data)
    source_data = source_data.transpose([2, 0, 1])
    source_data = np.expand_dims(source_data, axis=1)
    source_data = source_data[:2000, ...]
    # target_data
    file_str = r'data_for_training\kerry_cut.npy'
    target_data = np.load(file_str).astype(np.float32).transpose([2, 0, 1])
    target_data = np.expand_dims(target_data, axis=1)
    target_data = target_data[:2000, ...]
    print(f'source{source_data.shape}\ntarget{target_data.shape}')

    for i in range(target_data.shape[0]):
        temp = target_data[i, 0, :, :]
        temp = (temp - temp.mean()) / temp.std()
        temp = (temp - temp.min()) / (temp.max() - temp.min())  # 归一化到 [0, 1]
        target_data[i, 0, :, :] = temp

    source_data = torch.from_numpy(source_data)
    target_data = torch.from_numpy(target_data)

    # data set
    data_set = Data.TensorDataset(source_data, target_data)

    return data_set


def his_dataset_SYN2KERRYori():
    # source_labels
    file_str = r'data_for_training\FFEunconformity.npy'
    source_data = np.load(file_str).astype(np.float32)
    source_data = np.expand_dims(source_data, axis=1)
    source_data = source_data[:2000, ...]

    file_str = r'data_for_training\kerry_cut.npy'
    target_data = np.load(file_str).astype(np.float32).transpose([2, 0, 1])
    target_data = np.expand_dims(target_data, axis=1)
    target_data = target_data[:2000, ...]
    print(f'source{source_data.shape}\ntarget{target_data.shape}')

    source_data = source_data[:target_data.shape[0], ...]

    for i in range(target_data.shape[0]):
        temp = target_data[i, 0, :, :]
        temp = (temp - temp.mean()) / temp.std()
        temp = (temp - temp.min()) / (temp.max() - temp.min())
        target_data[i, 0, :, :] = temp

    source_data = torch.from_numpy(source_data)
    target_data = torch.from_numpy(target_data)

    # data set
    data_set = Data.TensorDataset(source_data, target_data)

    return data_set
