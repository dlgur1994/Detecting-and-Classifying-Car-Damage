# Importing Libraries
import torch
import os
import numpy as np


# Defining Data Loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        lst_img = os.listdir(self.img_dir)
        lst_label = os.listdir(self.label_dir)

        lst_img = [f for f in lst_img]
        lst_label = [f for f in lst_label]

        lst_img.sort()
        lst_label.sort()

        self.lst_img = lst_img
        self.lst_label = lst_label

    def __len__(self):
        return len(self.lst_img)

    def __getitem__(self, index):
        img = np.load(os.path.join(self.img_dir, self.lst_img[index]))
        label = np.load(os.path.join(self.label_dir, self.lst_label[index]))

        img = img / 255.0
        label = label / 255.0

        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        data = {'img': img, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


# Changing Data to Tensor
class ToTensor(object):
    def __call__(self, data):
        img, label = data['img'], data['label']

        img = img.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)

        data = {'img': torch.from_numpy(img), 'label': torch.from_numpy(label)}

        return data

# Applying Normalization on Data
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        img, label = data['img'], data['label']
        img = (img - self.mean) / self.std
        data = {'img': img, 'label': label}
        return data


# Randomizing Data Order
class RandomFlip(object):
    def __call__(self, data):
        img, label = data['img'], data['label']

        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)

        if np.random.rand() > 0.5:
            img = np.flipud(img)
            label = np.flipud(label)

        data = {'img': img, 'label': label}

        return data


# # Testing Data Loader
# import matplotlib.pyplot as plt

# # Sampling
# dataset_train = Dataset(img_dir='./datasets_npy/dent/test/images', label_dir='./datasets_npy/dent/test/masks')
# data = dataset_train.__getitem__(0)

# img = data['img']
# label = data['label']

# # Showing Sample
# plt.subplot(121)
# plt.imshow(img.squeeze())

# plt.subplot(122)
# plt.imshow(label.squeeze())

# plt.show()