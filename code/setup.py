import os

###
# Creates seperate subfolders for the validation dataset images based on their label keys from the val_annotations txt file
#
# Params:
#   directory: The directory path of the data folder in relation to the code folder
#
# Returns:
#    None
###
def setup(directory):

    imgDir = os.path.join(directory, 'val/images')
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
            os.rename(os.path.join(imgDir, img), os.path.join(newpath, img))