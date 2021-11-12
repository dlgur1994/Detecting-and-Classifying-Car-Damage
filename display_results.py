import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# Parsing Input
parser = argparse.ArgumentParser(description="Show the Result",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--img_dir", default="./results/numpy", type=str, dest="result_dir")
args = parser.parse_args()

result_dir = args.result_dir

lst_data = os.listdir(result_dir)

lst_img = [f for f in lst_data if f.startswith('img')]
lst_label = [f for f in lst_data if f.startswith('label')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_img.sort()
lst_label.sort()
lst_output.sort()


# Showing Sample
id = 0

img = np.load(os.path.join(result_dir, lst_img[id]))
label = np.load(os.path.join(result_dir, lst_label[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

plt.subplot(131)
plt.imshow(img)
plt.title('image')

plt.subplot(132)
plt.imshow(label)
plt.title('label')

plt.subplot(133)
plt.imshow(output)
plt.title('output')

plt.show()