import os
import random
import numpy as np

###
# Creates seperate subfolders for the validation dataset images based on their label keys from the val_annotations txt file
#
# Params:
#   directory: The directory path of the data folder in relation to the code folder
#
# Returns:
#    None
###

CLASSES = 5
IMG_PER_CLASS = 10


classes = np.random.choice(os.listdir("../data/train"), size=CLASSES, replace=False)

image_dict = {}
for img_class in classes:
    image = np.random.choice(os.listdir("../data/train/%s/images"%(img_class)), size=IMG_PER_CLASS, replace=False)
    image_dict[img_class] = image
    
newpath = "../MapCap_partition"
if not os.path.exists(newpath):
    os.makedirs(newpath)

classDict = {}
with open('../data/words.txt') as file:
    data = file.readlines()

    for class_ in classes:
        for line in data:
            words = line.split('\t')
            if class_ == words[0]:
                classDict[class_] = words[1].split(',')[0].strip()
    
for cl, images in image_dict.items():
    for image in images:
        old_path = os.path.join("../data/train/%s/images"%(cl), image)
        label = classDict[cl]
        os.makedirs(os.path.join(newpath,label), exist_ok=True)
        updated_path = os.path.join(newpath, label, image)
        os.rename(old_path, updated_path)



"""def setup(directory):

    imgDir = os.path.join(directory, 'train')

    # Open and read val annotations text file
    with open(os.path.join(directory, 'val/val_annotations.txt'), 'r') as filePath:
        data = filePath.readlines()

        # Creates a dictionary to store img filename and corresponding class name
        imgDict = {}
        for line in data:
            words = line.split('\t')
            imgDict[words[0]] = words[1]

    # Create subfolders and moves images into their respective folders
    for img, name in imgDict.items():
        newpath = (os.path.join(imgDir, name))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(imgDir, img)):
            os.rename(os.path.join(imgDir, img), os.path.join(newpath, img))"""