import os
import copy
import torch.utils.data as data
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
import numpy as np
import pandas as pd


def insert_outliers(fine, coarse, labels, outlier_ratio=0.1, outlier_range=(1, 4)):

    length = len(fine)
    for i in range(length):
        outlier_num = int(fine[i].shape[0] * outlier_ratio)
        outlier_index = np.random.choice(fine[i].shape[0], outlier_num, replace=False)
        outlier_value = np.random.uniform(outlier_range[0], outlier_range[1], (outlier_num, fine[i].shape[1]))
        fine[i][outlier_index] += outlier_value

    outlier_num = int(coarse.shape[0] * outlier_ratio)
    outlier_index = np.random.choice(coarse.shape[0], outlier_num, replace=False)
    outlier_value = np.random.uniform(outlier_range[0], outlier_range[1], (outlier_num, coarse.shape[1]))
    coarse[outlier_index] += outlier_value

    for i in range(len(fine)):
        sigma = outlier_ratio * 0.2
        clip = sigma * 2
        labels[i] += np.clip(sigma * np.random.randn(*labels[i].shape), -1 * clip, clip)
    return fine, coarse, labels


def data_jitter(fine, coarse, label, sigma=0.01):
    # for i in range(int(len(fine)/3), len(fine)):
    clip = sigma * 2
    for i in range(len(fine)):
        fine[i] += np.clip(sigma * np.random.randn(*fine[i].shape), -1 * clip, clip)
        coarse[i] += np.clip(sigma * np.random.randn(*coarse[i].shape), -1 * clip, clip)
        label[i] += np.clip(sigma * np.random.randn(*label[i].shape), -1 * clip, clip)
    return fine, coarse, label


def data_augmentation(fine, coarse, labels, drop=0.1):
    length = len(fine)
    # swap xy (mirror transformation)
    for i in range(length):
        fine.append(copy.deepcopy(fine[i][:, [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]]))
        # phi不变 theta = (0.9 - theta + 3.6) mod 3.6
        fine[-1][:, 8] = (0.9 - fine[-1][:, 8] + 3.6) % 3.6
        coarse = np.append(coarse, coarse[i].reshape(1, -1), axis=0)
        labels = np.append(labels, labels[i].reshape(1, -1), axis=0)

    # central symmetry transformation (7, 7, 7)
    for i in range(length):
        fine.append(copy.deepcopy(fine[i]))
        fine[-1][:, :3] = 14 - fine[-1][:, :3]
        # phi unchanged, theta = (theta + 1.8) mod 3.6
        fine[-1][:, 8] = (fine[-1][:, 8] + 1.8) % 3.6
        coarse = np.append(coarse, coarse[i].reshape(1, -1), axis=0)
        labels = np.append(labels, labels[i].reshape(1, -1), axis=0)

    # Randomly drop some data points
    if drop > 0:
        for i in range(len(fine)):
            # Delete some data points
            drop_num = int(fine[i].shape[0] * drop)
            drop_index = np.random.choice(fine[i].shape[0], drop_num, replace=False)
            fine[i] = np.delete(fine[i], drop_index, axis=0)

    return fine, coarse, labels


def fill_and_sample(fines, num=1024, fill='zero'):
    for i, fine in enumerate(fines):
        if fine.shape[0] >= num:
            sample = np.random.choice(fine.shape[0], num, replace=False)
            fine = fine[sample]
        else:
            add_dim = num - fine.shape[0]
            if fill == 'zero':
                zero_np = np.zeros((add_dim, fine.shape[1]))
                fine = np.concatenate((fine, zero_np), axis=0)
            elif fill == 'repeat':
                repeat_np = np.repeat(fine, add_dim // fine.shape[0] + 1, axis=0)
                fine = np.concatenate((fine, repeat_np[:add_dim]), axis=0)
        fines[i] = fine
    fine = np.array(fines, dtype=np.float32)
    return fine


def data_normalize(fine, coarse, labels, label_norm, part, root):
    for i in range(len(fine)):  # 6 columns anisotropy, 9 columns sphericity, no normalization
        # Columns 0 to 2 are coordinates, divide by 14;
        fine[i][:, :3] /= 14
        # Columns 4th and 5th are volume and surface area; apply log(x + 1) normalization with base 200.
        fine[i][:, 4:6] = np.log1p(fine[i][:, 4:6]) / np.log(200)
        # 7th column is phi, divide by 0.9; 8th column is theta in [0, 3.6], divide by 3.6
        fine[i][:, 7] /= 0.9
        fine[i][:, 8] /= 3.6

    # fine 3rd column is EqDiameter(mm), using minmax normalization; coarse using minmax normalization
    if part == 'train':
        feature_scaler = MinMaxScaler()
        all_fine_3 = np.concatenate([x[:, 3].reshape(-1, 1) for x in fine], axis=0)
        feature_scaler.fit(all_fine_3)
        for i in range(len(fine)):
            fine[i][:, 3] = feature_scaler.transform(fine[i][:, 3].reshape(-1, 1)).flatten()
        global_scaler = MinMaxScaler()
        coarse = global_scaler.fit_transform(coarse)
        scalers = {
            'scaler_fine': feature_scaler,
            'scaler_coarse': global_scaler
        }
        joblib.dump(scalers, 'dataset/scalers.bin')
    else:
        assert os.path.exists('dataset/scalers.bin'), 'scalers.bin not found, run train dataset first'
        scalers = joblib.load('dataset/scalers.bin')
        for i in range(len(fine)):
            fine[i][:, 3] = scalers['scaler_fine'].transform(fine[i][:, 3].reshape(-1, 1)).flatten()
        coarse = scalers['scaler_coarse'].transform(coarse)

    if label_norm:
        # log normalization
        min_vals = np.min(labels, axis=1, keepdims=True)
        max_vals = np.max(labels, axis=1, keepdims=True)
        labels = (np.log(labels) - np.log(min_vals)) / (np.log(max_vals) - np.log(min_vals))
        if part == 'test':
            joblib.dump({
                'min_vals': min_vals,
                'max_vals': max_vals
            }, Path(root) / 'min_max_vals.bin')

    return fine, coarse, labels


class FoamAlDataset(data.Dataset):
    def __init__(self, root=Path('dataset/'), label_norm=True, part='train', npoints=1024, augmentation=True,
                 drop=0.1, jitter=0, outlier_ratio=0.1):
        self.root = Path(root) / part
        fine_path = self.root / 'local'
        coarse_path = self.root / "global.csv"
        curve_path = self.root / "energy"

        # average cell thickness, fractal dimension, porosity
        coarse = np.array(pd.read_csv(str(coarse_path)).values[:, [1, 2, 3]], dtype=np.float32)  # (M, 3)
        # energy absorption curve, 50 points
        labels = []
        for i in curve_path.iterdir():
            label = np.array(pd.read_csv(str(i)).values[:, 1], dtype=np.float32)
            labels.append(label)
        labels = np.array(labels, dtype=np.float32)     # (M, 50)  
        # feature point cloud
        fine = []
        for i in fine_path.iterdir():
            x = pd.read_csv(str(i)).values
            fine.append(x[:, :10])  # (M, N?, 10)

        # Convert 8th column theta = (theta + 3.6) mod 3.6, range from (-1.8, 1.8) to (0, 3.6)
        for i in range(len(fine)):
            fine[i][:, 8] = (fine[i][:, 8] + 3.6) % 3.6

        # data augmentation
        if part == 'train' and augmentation:
            fine, coarse, labels = data_augmentation(fine, coarse, labels, drop=drop)

        # normalization
        fine, coarse, labels = data_normalize(fine, coarse, labels, label_norm, part, root)

        # data perturbation
        if part == 'train' and augmentation and jitter > 0:
            fine, coarse, labels = data_jitter(fine, coarse, labels, sigma=jitter)
        if part == 'train' and augmentation and outlier_ratio > 0:
            fine, coarse, labels = insert_outliers(fine, coarse, labels, outlier_ratio=outlier_ratio)

        # fill and sample
        fine = fill_and_sample(fine, npoints, fill='zero')  # (M, N, 10)

        # transfer to tensor
        self.fine = torch.from_numpy(fine)
        self.coarse = torch.from_numpy(coarse)
        self.labels = torch.from_numpy(labels)
        self.f_channel = self.fine.shape[2]
        self.c_channel = self.coarse.shape[1]
        self.label_channel = self.labels.shape[1]

    def __getitem__(self, index):
        # (M, N, 10)
        # (M, N, 3)
        # (M, N, 50)
        return self.fine[index], self.coarse[index], self.labels[index]

    def __len__(self):
        return len(self.fine)


if __name__ == '__main__':
    train_dataset = FoamAlDataset(part='train')
    print(train_dataset[0])
    test_dataset = FoamAlDataset(part='test')
    print(test_dataset[0])
    print(train_dataset[0][0].shape, train_dataset[0][1].shape, train_dataset[0][2].shape)
