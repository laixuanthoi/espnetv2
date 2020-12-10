import cv2
import glob
import os
import random
import numpy as np

# save_dir = "lane_segmentation"
data_dir = "segmentation/data"
label_dir = "segmentation/label"

train_file = "train.txt"
val_file = "val.txt"

fileNames = []
for f in glob.glob(os.path.join(label_dir, "*.jpg")):
    basename = os.path.basename(f)
    fileNames.append(basename)

np.random.shuffle(fileNames)

rate = int(len(fileNames)*0.7)

train_arr = fileNames[:rate]
val_arr = fileNames[rate:]

def writeTextFile(textFile, arr):
    with open(textFile, 'w') as f:
        for a in arr:
            f.write('segmentation/data/{},segmentation/label/{}\n'.format(a, a))

writeTextFile(train_file, train_arr)
writeTextFile(val_file, val_arr)