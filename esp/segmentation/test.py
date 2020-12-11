import glob
from cnn import SegmentationModel as net
from torchsummary import summary
import torch
import cv2
import numpy as np
import os

PATH = "results_espnetv2_1/model_best.pth"


def image_feed(image):
    img = cv2.resize(image, (224, 224))
    h, w = img.shape[:2]
    img = img.astype(np.float32)
    img /= 255
    img = img.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
    img_variable = torch.autograd.Variable(img_tensor, volatile=True)
    return img_variable, h, w


checkpoint = torch.load(PATH)
# print(checkpoint)
model = net.EESPNet_Seg(3, s=2)
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
model.load_state_dict(checkpoint, strict=False)
model.eval()


for f in glob.glob(os.path.join("dataset/data/*.png")):
    image = cv2.imread(f)

    input_img, H_ori, W_ori = image_feed(image)

    # print(input_img.size())
    # input_var = torch.autograd.Variable(input_img, volatile=True)
    input_img.cuda()
    img_out = model(input_img)

    pallete = [
        [0, 0, 0],
        [0, 0, 255],
        [255, 255, 255]
    ]

    classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()

    print(classMap_numpy)

    classMap_numpy_color = np.zeros(
        (224, 224, 3), dtype=np.uint8)

    for idx in range(len(pallete)):
        print(idx)
        [r, g, b] = pallete[idx]
        classMap_numpy_color[classMap_numpy == idx] = [b, g, r]

    classMap_numpy_color = cv2.resize(classMap_numpy_color, (320, 120))
    cv2.imshow("color", classMap_numpy_color)
    cv2.imshow("image", image)
    cv2.waitKey(0)
