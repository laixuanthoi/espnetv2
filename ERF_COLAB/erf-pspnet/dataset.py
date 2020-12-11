import glob
import cv2
import os
import numpy as np

# for f in glob.glob(os.path.join("dataset", "train", "*.jpg")):
#     print(f)
#     basename = os.path.basename(f)
#     filename = os.path.splitext(basename)[0]
#     img = cv2.imread(f)
#     cv2.imwrite(os.path.join("normdataset", "train",
#                              "{}.png".format(filename)), img)

# # trainannot
# # valannot
# for f in glob.glob(os.path.join("dataset", "trainannot", "*.png")):
#     basename = os.path.basename(f)
#     filename = os.path.splitext(basename)[0]
#     img = cv2.imread(f, 0)

#     # newImg = np.zeros(img.shape, np.uint8)

#     # b_channel = img[:,:,0]
#     # g_channel = img[:,:,1]
#     # r_channel = img[:,:,2]

#     # newImg[b_channel == 255 & b_channel == 255] = 255

#     tmp_img = img.copy()
#     img[(tmp_img >= 10) & (tmp_img < 100)] = 1  # blue
#     img[tmp_img > 200] = 2  # blue
#     img[tmp_img < 10] = 0  # blue

#     cv2.imshow('image', img)
#     # cv2.imshow('newImg', newImg)
#     # cv2.waitKey(0)
#     cv2.imwrite(os.path.join("normdataset", "trainannot",
#                              "{}.png".format(filename)), img)
