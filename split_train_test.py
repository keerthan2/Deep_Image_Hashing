import os
import numpy as np
import shutil
import random

# # Creating Train / Val / Test folders (One time use)
root_dir = '../grocery_data'
root_dir2 = '../grocery_data_split'
classes_dir = os.listdir(root_dir)

val_ratio = 0.25
test_ratio = 0.05

if not os.path.exists(root_dir2):
  os.mkdir(root_dir2)

for cls in classes_dir:
    os.makedirs(os.path.join(root_dir2,'train',cls))
    os.makedirs(os.path.join(root_dir2,'val',cls))
    os.makedirs(os.path.join(root_dir2,'test',cls))


    # Creating partitions of the data after shuffeling
    src = os.path.join(root_dir,cls) # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)* (1 - val_ratio + test_ratio)), 
                                                               int(len(allFileNames)* (1 - test_ratio))])


    train_FileNames = [os.path.join(src,name) for name in train_FileNames.tolist()]
    val_FileNames = [os.path.join(src,name) for name in val_FileNames.tolist()]
    test_FileNames = [os.path.join(src,name) for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, os.path.join(root_dir2,'train',cls))

    for name in val_FileNames:
        shutil.copy(name, os.path.join(root_dir2,'val',cls))

    for name in test_FileNames:
        shutil.copy(name, os.path.join(root_dir2,'test',cls))