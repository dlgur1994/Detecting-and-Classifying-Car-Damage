# Importing Libraries
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

from model import UNet
from efficientunet import *

from dataset import *
from util import *


# Parsing Inputs
parser = argparse.ArgumentParser(description="Test the Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model_name", default="UNet", type=str, dest="model_name")

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")

parser.add_argument("--data_dir", default="./datasets_npy", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

args = parser.parse_args()


# Setting Variables
model_name = args.model_name

lr = args.lr
batch_size = args.batch_size

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
result_dir = args.result_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOOTH = 1e-6

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
if model_name == 'UNet':
    net = UNet().to(device)
elif model_name == 'efficientunet_b0':
    net = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True).to(device)
elif model_name == 'efficientunet_b1':
    net = get_efficientunet_b1(out_channels=1, concat_input=True, pretrained=True).to(device)
elif model_name == 'efficientunet_b2':
    net = get_efficientunet_b2(out_channels=1, concat_input=True, pretrained=True).to(device)
elif model_name == 'efficientunet_b3':
    net = get_efficientunet_b3(out_channels=1, concat_input=True, pretrained=True).to(device)
elif model_name == 'efficientunet_b4':
    net = get_efficientunet_b4(out_channels=1, concat_input=True, pretrained=True).to(device)
elif model_name == 'efficientunet_b5':
    net = get_efficientunet_b5(out_channels=1, concat_input=True, pretrained=True).to(device)
elif model_name == 'efficientunet_b6':
    net = get_efficientunet_b6(out_channels=1, concat_input=True, pretrained=True).to(device)
elif model_name == 'efficientunet_b7':
    net = get_efficientunet_b7(out_channels=1, concat_input=True, pretrained=True).to(device)


# Defining Optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)


# Test Mode
# defining functions for saving outputs
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > -0.1)
'''
UNet
  - dent: -1.1(0.6221), scratch: -1.3(0.4251), spacing: -0.6(0.4498)
Efb0
  - dent: -1.5(0.6146), scratch: (), spacing: ()
Efb1
  - dent: 0(0.6322), scratch: (), spacing: ()
Efb2
  - dent: (), scratch: (), spacing: ()
Efb3
  - dent: (), scratch: (), spacing: ()
Efb4
  - dent: 1.2(0.6370), scratch: -0.1(0.5308), spacing: 0.6(0.4471)
Efb5
  - dent: (), scratch: (), spacing: ()
Efb6
  - dent: (), scratch: (), spacing: ()
Efb7
  - dent: (), scratch: (), spacing: ()
'''

net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad():
    net.eval()
    iou_scores = []

    for batch, data in enumerate(loader_test, 1):
        # forward pass
        img = data['img'].to(device)
        label = data['label'].to(device)
        output = net(img)
        
        # saving outputs
        img = fn_tonumpy(fn_denorm(img, mean=0.5, std=0.5))
        label = fn_tonumpy(label)
        output = fn_tonumpy(fn_class(output))

        # for j in range(label.shape[0]):
        #     id = num_batch_test * (batch - 1) + j

        #     plt.imsave(os.path.join(result_dir, 'jpg', 'image_%04d.png' % id), img[j].squeeze(), cmap='gray')
        #     plt.imsave(os.path.join(result_dir, 'jpg', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
        #     plt.imsave(os.path.join(result_dir, 'jpg', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

        #     np.save(os.path.join(result_dir, 'numpy', 'image_%04d.npy' % id), img[j].squeeze())
        #     np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
        #     np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

        label_temp = label.squeeze(3).astype(int)
        output_temp = output.squeeze(3).astype(int)

        intersection = np.sum(label_temp & output_temp)
        union = np.sum(label_temp | output_temp)
        iou_score = (intersection + SMOOTH) / (union + SMOOTH)

        iou_scores.append(iou_score)
        print("TEST: BATCH %04d \ %04d | IoU %.4f" % (batch, num_batch_test, iou_score))

print("AVERAGE TEST: BATCH %04d \ %04d | mIoU %.4f" % (batch, num_batch_test, sum(iou_scores)/len(iou_scores)))