import cv2
import numpy as np


image = cv2.imread("segmentation/label/frame_1.jpg", 0)
cv2.imshow("image", image)


blue, white = image.copy(), image.copy()
blue[(blue > 20) & (blue < 40)] = 255

cv2.imshow("blue", blue)

unique = np.unique(image)
print(unique)

cv2.waitKey(0)