# Importing Libraries
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import UNet
from dataset import *
from util import *

import argparse


# Parsing Inputs
parser = argparse.ArgumentParser(description="Train the Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets_npy", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()


# Setting Variables
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Making Directories
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'jpg'))
    os.makedirs(os.path.join(result_dir, 'numpy'))


# Training Network
if mode == "train":
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    
    dataset_train_dent = Dataset(img_dir=os.path.join(data_dir, 'dent', 'train', 'images'), \
                                 label_dir=os.path.join(data_dir, 'dent', 'train', 'masks'), \
                                 transform=transform)
    dataset_train_scratch = Dataset(img_dir=os.path.join(data_dir, 'scratch', 'train', 'images'), \
                                    label_dir=os.path.join(data_dir, 'scratch', 'train', 'masks'), \
                                    transform=transform)
    dataset_train_spacing = Dataset(img_dir=os.path.join(data_dir, 'spacing', 'train', 'images'), \
                                    label_dir=os.path.join(data_dir, 'spacing', 'train', 'masks'), \
                                    transform=transform)
    dataset_train = dataset_train_dent + dataset_train_scratch + dataset_train_spacing

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val_dent = Dataset(img_dir=os.path.join(data_dir, 'dent', 'valid', 'images'), \
                               label_dir=os.path.join(data_dir, 'dent', 'valid', 'masks'), \
                               transform=transform)
    dataset_val_scratch = Dataset(img_dir=os.path.join(data_dir, 'scratch', 'valid', 'images'), \
                                  label_dir=os.path.join(data_dir, 'scratch', 'valid', 'masks'), \
                                  transform=transform)
    dataset_val_spacing = Dataset(img_dir=os.path.join(data_dir, 'spacing', 'valid', 'images'), \
                                  label_dir=os.path.join(data_dir, 'spacing', 'valid', 'masks'), \
                                  transform=transform)
    dataset_val = dataset_val_dent + dataset_val_scratch + dataset_val_spacing

    loader_val = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test_dent = Dataset(img_dir=os.path.join(data_dir, 'dent', 'test', 'images'), \
                                 label_dir=os.path.join(data_dir, 'dent', 'test', 'masks'), \
                                 transform=transform)
    dataset_test_scratch = Dataset(img_dir=os.path.join(data_dir, 'scratch', 'test', 'images'), \
                                    label_dir=os.path.join(data_dir, 'scratch', 'test', 'masks'), \
                                    transform=transform)
    dataset_test_spacing = Dataset(img_dir=os.path.join(data_dir, 'spacing', 'test', 'images'), \
                                    label_dir=os.path.join(data_dir, 'spacing', 'test', 'masks'), \
                                    transform=transform)
    dataset_test = dataset_test_dent + dataset_test_scratch + dataset_test_spacing

    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)


# Creating Network
net = UNet().to(device)


# Defining Loss Function
fn_loss = nn.BCEWithLogitsLoss().to(device)


# Defining Optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)


# Defining Variables
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


# Training
st_epoch = 0

# Train Mode
if mode == "train":
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            img = data['img'].to(device)
            label = data['label'].to(device)

            output = net(img)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # calculate loss function
            loss_arr += [loss.item()]

            print("Train: EPOCH %04d / %04d | BATCH %04d \ %04d | LOSS %.4f"
                  % (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # saving Tensorboard
            img = fn_tonumpy(fn_denorm(img, mean=0.5, std=0.5))
            label = fn_tonumpy(label)
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('img', img, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                img = data['img'].to(device)
                label = data['label'].to(device)

                output = net(img)

                # calculate loss function
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("Validation: EPOCH %04d / %04d | BATCH %04d \ %04d | LOSS %.4f"
                      % (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # saving Tensorboard
                img = fn_tonumpy(fn_denorm(img, mean=0.5, std=0.5))
                label = fn_tonumpy(label)
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('img', img, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 1 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# Test Mode
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            img = data['img'].to(device)
            label = data['label'].to(device)

            output = net(img)

            # calculate loss function
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d \ %04d | LOSS %.4f"
                  % (batch, num_batch_test, np.mean(loss_arr)))

            # saving Tensorboard
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

    print("AVERAGE TEST: BATCH %04d \ %04d | LOSS %.4f"
          % (batch, num_batch_test, np.mean(loss_arr)))