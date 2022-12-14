import os
import shutil
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

CLASSES = 25
IMG_PER_CLASS = 100


classes = np.random.choice(os.listdir("../tiny-imagenet-200/train"), size=CLASSES, replace=False)

image_dict = {}
for img_class in classes:
    image = np.random.choice(os.listdir("../tiny-imagenet-200/train/%s/images"%(img_class)), size=IMG_PER_CLASS, replace=False)
    image_dict[img_class] = image
    
newpath = "../MapCap_partition"
if os.path.exists(newpath):
    shutil.rmtree(newpath)
os.makedirs(newpath)

classDict = {}
with open('../tiny-imagenet-200/words.txt') as file:
    data = file.readlines()

    for class_ in classes:
        for line in data:
            words = line.split('\t')
            if class_ == words[0]:
                classDict[class_] = words[1].split(',')[0].strip()
    
for cl, images in image_dict.items():
    for image in images:
        old_path = os.path.join("../tiny-imagenet-200/train/%s/images"%(cl), image)
        label = classDict[cl]
        os.makedirs(os.path.join(newpath,label), exist_ok=True)
        updated_path = os.path.join(newpath, label, image)
        os.rename(old_path, updated_path)
