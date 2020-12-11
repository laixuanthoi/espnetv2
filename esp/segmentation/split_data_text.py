import cv2
import glob
import os
import random
import numpy as np

# save_dir = "lane_segmentation"
data_dir = "dataset/data"
label_dir = "dataset/label"

train_file = "dataset/train.txt"
val_file = "dataset/val.txt"

fileNames = []
for f in glob.glob(os.path.join(label_dir, "*.png")):
    basename = os.path.basename(f)
    fileNames.append(basename)

np.random.shuffle(fileNames)

rate = int(len(fileNames)*0.7)

train_arr = fileNames[:rate]
val_arr = fileNames[rate:]


def writeTextFile(textFile, arr):
    with open(textFile, 'w') as f:
        for a in arr:
            f.write('dataset/data/{},dataset/label/{}\n'.format(a, a))


writeTextFile(train_file, train_arr)
writeTextFile(val_file, val_arr)
