# Importing Libraries
import os
from PIL import Image
import numpy as np


# Setting Paths
dir_data = './datasets'
dir_data_new = './datasets_gray_npy'
category_damages = ['dent', 'scratch', 'spacing']
category_data = ['test', 'train', 'valid']
category_represents = ['images', 'masks']


# Making Directories and Changing to npz files
for damage in category_damages:
    for data in category_data:
        for represent in category_represents:
            if not os.path.exists(os.path.join(dir_data_new, damage, data, represent)):
                os.makedirs(os.path.join(dir_data_new, damage, data, represent))

            cnt = 0
            for file in os.listdir(os.path.join(dir_data, damage, data, represent)):
                if represent == 'images':
                    data_temp = Image.open(os.path.join(dir_data, damage, data, represent, file))
                    data_final = data_temp.convert('L').resize((512, 512))

                else:
                    data_temp = Image.open(os.path.join(dir_data, damage, data, represent, file)).convert('L')
                    data_temp = np.array(data_temp)

                    for i in range(data_temp.shape[0]):
                        for j in range(data_temp.shape[1]):
                            if data_temp[i][j] < 10:
                                data_temp[i][j] = 0
                            else:
                                data_temp[i][j] = 255

                    data_final = np.array(Image.fromarray(data_temp).resize((512, 512), Image.NEAREST))

                np.save(os.path.join(dir_data_new, damage, data, represent, file.split('.jpg')[0] + '.npy'),
                        data_final)
                cnt += 1
            print('{0}_{1}_{2} done'.format(damage, data, represent))
print('Coverting Done!')


# # Showing Samples
# img = np.load('./datasets_gray_npy/dent/test/images/20190412_9677_21799705_743ea43c1564a46ef68ddff5a1e77fd7.npy')
# mask = np.load('./datasets_gray_npy/dent/test/masks/20190412_9677_21799705_743ea43c1564a46ef68ddff5a1e77fd7.npy')
#
# img_array = Image.fromarray(img)
# mask_array = Image.fromarray(mask)
#
# print(np.unique(img))
# print(np.unique(mask))
#
# img_array.show()
# mask_array.show()
