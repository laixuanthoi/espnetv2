import cv2
import os
import glob
import numpy as np

label_dir = "my_data/label"
data_dir = "my_data/data"


fileNames = []
for f in glob.glob(os.path.join(label_dir, "*.jpg")):
    basename = os.path.basename(f)
    fileNames.append(basename)

np.random.shuffle(fileNames)

rate = int(len(fileNames)*0.7)

train_arr = fileNames[:rate]
val_arr = fileNames[rate:]


for f in train_arr:
    filename = os.path.splitext(f)[0]
    rgb_img = cv2.imread(os.path.join("my_data", "data", f))
    print(rgb_img.shape[:2])
    label_img = cv2.imread(os.path.join("my_data", "label", f))
    cv2.imwrite(os.path.join("dataset", "train", "{}.jpg".format(filename)), rgb_img)
    cv2.imwrite(os.path.join("dataset", "trainannot", "{}.png".format(filename)), label_img)


for f in val_arr:
    filename = os.path.splitext(f)[0]
    rgb_img = cv2.imread(os.path.join("my_data", "data", f))
    label_img = cv2.imread(os.path.join("my_data", "label", f))
    cv2.imwrite(os.path.join("dataset", "val", "{}.jpg".format(filename)), rgb_img)
    cv2.imwrite(os.path.join("dataset", "valannot", "{}.png".format(filename)), label_img)