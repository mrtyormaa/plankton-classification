__author__ = 'mrtyormaa'

import glob
import os

from PIL import Image
# im = Image.open("435.jpg")
# im.rotate(180).save("435-rotated" + ".jpg", "JPEG")

# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("train-all-data", "*"))
                           ).difference(set(glob.glob(os.path.join("train-all-data", "*.*")))))

# Rescale the images and create the combined metrics and training labels
# get the total training images
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
                continue
            numberofImages += 1
            # print(os.path.splitext(fileName)[0])
            fullName = fileNameDir[0] + '/' + fileName
            newName = fileNameDir[0] + '/' + os.path.splitext(fileName)[0] + '-rotated.jpg'
            im = Image.open(fullName)
            im.rotate(180).save(newName, "JPEG")
