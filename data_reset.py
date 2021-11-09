## Importing Libraries
import os
import shutil


## Setting Paths
dir_data = './accida_segmentation_dataset_v1'
dir_data_new = './datasets'
category_damages = ['dent', 'scratch', 'spacing']
category_data = ['test', 'train', 'valid']
category_represent = ['images', 'masks']


## Making Directories
for damage in category_damages:
    for data in category_data:
        for represent in category_represent:
            if not os.path.exists(os.path.join(dir_data_new, damage, data, represent)):
                os.makedirs(os.path.join(dir_data_new, damage, data, represent))


## Saving Data
for damage in category_damages:
    for data in category_data:
        for represent in category_represent:
            cnt = 0
            for file in os.listdir(os.path.join(dir_data, damage, data, represent)):
                if 'augmented' in file:
                    continue
                shutil.copyfile(os.path.join(dir_data, damage, data, represent, file),
                            os.path.join(dir_data_new, damage, data, represent, file))
                cnt += 1
            print('{0}_{1}_{2}: {3}'.format(damage, data, represent, cnt))
print('Copy done!')