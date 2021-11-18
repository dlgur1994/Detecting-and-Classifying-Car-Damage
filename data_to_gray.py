# Importing Libraries
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Setting Paths
dir_data = './datasets'
dir_data_new = './datasets_gray'
category_damages = ['dent', 'scratch', 'spacing']
category_data = ['test', 'train', 'valid']
category_represents = ['images', 'masks']


# Making Directories and Changing to npz files
for damage in category_damages:
    for data in category_data:
        for represent in category_represents:
            if not os.path.exists(os.path.join(dir_data_new, damage, data, represent)):
                os.makedirs(os.path.join(dir_data_new, damage, data, represent))

            for e in os.listdir(os.path.join(dir_data, damage, data, represent)):
                data_temp = Image.open(os.path.join(dir_data, damage, data, represent, e))
                data_final = data_temp.convert('L')
                data_final.save(os.path.join(dir_data_new, damage, data, represent, e))
            print('{0}_{1}_{2} done'.format(damage, data, represent))
print('Coverting Done!')


# # Showing Samples
# img_array = Image.open('./datasets_gray/dent/test/images/20180101_4052_10643410_3397b1717a1b2f8e34c95d945fc14938.jpg')
# mask_array = Image.open('./datasets_gray/dent/test/masks/20180101_4052_10643410_3397b1717a1b2f8e34c95d945fc14938.jpg')
#
# plt.subplot(121)
# plt.imshow(img_array, cmap='gray')
# plt.title('image')
#
# plt.subplot(122)
# plt.imshow(mask_array, cmap='gray')
# plt.title('mask')
#
# plt.show()
#
# print(img_array.size)
# print(mask_array.size)