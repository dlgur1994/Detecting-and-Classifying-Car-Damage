# Importing Libraries
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

import wandb
wandb.init(project="U-Net", entity="kim1lee3")

from model import UNet
from dataset import *
from util import *


# Parsing Inputs
parser = argparse.ArgumentParser(description="Train the Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets_npy", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")

parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()


# Setting Variables
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir

train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.config = {
  "learning_rate": args.lr,
  "epochs": args.num_epoch,
  "batch_size": args.batch_size
}


# Network
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

dataset_train = Dataset(img_dir=os.path.join(data_dir, 'train', 'images'), \
                              label_dir=os.path.join(data_dir, 'train', 'masks'), \
                              transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val = Dataset(img_dir=os.path.join(data_dir, 'valid', 'images'), \
                            label_dir=os.path.join(data_dir, 'valid', 'masks'), \
                            transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)


# Creating Network
net = UNet().to(device)


# Defining Loss Function
fn_loss = nn.BCEWithLogitsLoss().to(device)


# Defining Optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)


# Train
st_epoch = 0

if train_continue == "on":
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_train = []

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
        loss_train += [loss.item()]
        print("Train: EPOCH %04d / %04d | BATCH %04d \ %04d | LOSS %.4f"
              % (epoch, num_epoch, batch, num_batch_train, np.mean(loss_train)))

    wandb.log({"train loss": np.mean(loss_train)})

    with torch.no_grad():
        net.eval()
        loss_valid = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            img = data['img'].to(device)
            label = data['label'].to(device)
            output = net(img)

            # calculate loss function
            loss = fn_loss(output, label)
            loss_valid += [loss.item()]
            print("Validation: EPOCH %04d / %04d | BATCH %04d \ %04d | LOSS %.4f"
                  % (epoch, num_epoch, batch, num_batch_val, np.mean(loss_valid)))

    wandb.log({"valid loss": np.mean(loss_valid)})

    if epoch % 10 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)