import cv2
import numpy as np
import glob
import os

LABEL_DIR = 'segmentation/label'
SAVE_DIR ='segmentation/newlabel'

for f in glob.glob(os.path.join(LABEL_DIR, '*.jpg')):
    basename = os.path.basename(f)
    image = cv2.imread(f, 0)
    blue, white = image.copy(), image.copy()
    image[(blue > 20) & (blue < 40)] = 1 #blue
    image[blue > 240] = 2 #white
    image[blue < 20] = 0
    cv2.imwrite(os.path.join(SAVE_DIR, basename), image)
    print(os.path.join(SAVE_DIR, basename))


image = cv2.imread('segmentation/newlabel/frame_144.jpg', 0)

unique = np.unique(image)
print(unique)

# cv2.waitKey(0)