# Importing Libraries
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

from model import UNet
from dataset import *
from util import *


# Parsing Inputs
parser = argparse.ArgumentParser(description="Test the Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")

parser.add_argument("--data_dir", default="./datasets_npy", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

args = parser.parse_args()


# Setting Variables
lr = args.lr
batch_size = args.batch_size

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
result_dir = args.result_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Making Directories
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'jpg'))
    os.makedirs(os.path.join(result_dir, 'numpy'))


# Setting
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

dataset_test = Dataset(img_dir=os.path.join(data_dir, 'test', 'images'), \
                       label_dir=os.path.join(data_dir, 'test', 'masks'), \
                       transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

num_data_test = len(dataset_test)

num_batch_test = np.ceil(num_data_test / batch_size)


# Creating Network
net = UNet().to(device)


# Defining Optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)


# Test Mode
# defining functions for saving outputs
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > -6.0)

net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad():
    net.eval()
    iou_scores = []

    for batch, data in enumerate(loader_test, 1):
        # forward pass
        img = data['img'].to(device)
        label = data['label'].to(device)
        output = net(img)

        # label_temp = label.squeeze(1)
        # output_temp = output.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
        # intersection = (output_temp & label_temp).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        # union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0    
        # iou_score = np.sum(intersection) / np.sum(union)
        
        # iou_scores.append(iou_score)
        # print("TEST: BATCH %04d \ %04d | IoU %.4f" % (batch, num_batch_test, iou_score))
        print("TEST: BATCH %04d \ %04d" % (batch, num_batch_test))
        
        # saving outputs
        img = fn_tonumpy(fn_denorm(img, mean=0.5, std=0.5))
        label = fn_tonumpy(label)
        output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            id = num_batch_test * (batch - 1) + j

            plt.imsave(os.path.join(result_dir, 'jpg', 'image_%04d.png' % id), img[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'jpg', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'jpg', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

            np.save(os.path.join(result_dir, 'numpy', 'image_%04d.npy' % id), img[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())
        # break
# print("AVERAGE TEST: BATCH %04d \ %04d | mIoU %.4f" % (batch, num_batch_test, np.mean(iou_scores)))