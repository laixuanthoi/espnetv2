import glob
import cv2
import os

# for f in glob.glob(os.path.join("dataset", "val", "*.jpg")):
#     print(f)
#     basename = os.path.basename(f)
#     filename = os.path.splitext(basename)[0]
#     img = cv2.imread(f)
#     cv2.imwrite(os.path.join("newdataset", "val", "{}.png".format(filename)), img)

for f in glob.glob(os.path.join("dataset", "valannot", "*.png")):
    basename = os.path.basename(f)
    filename = os.path.splitext(basename)[0]
    img = cv2.imread(f, 0)
    tmp_img = img.copy()
    img[(tmp_img > 20) & (tmp_img < 40)] = 50 #blue
    img[tmp_img > 240] = 255 #blue
    img[tmp_img < 20] = 0 #blue

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join("newdataset", "valannot", "{}.png".format(filename)), img)