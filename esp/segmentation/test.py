import glob
from cnn import SegmentationModel as net
from torchsummary import summary
import torch
import cv2
import numpy as np
import os
from time import time

PATH = "results_espnetv2_0.5/checkpoint.pth.tar"
# PATH = "espnetv2_s_0.5.pth"


def image_feed(image):
    mean = [133.49126, 123.00861, 120.21908]
    std = [41.028954, 41.625504, 43.69621]

    img = image.astype(np.float32)

    for j in range(3):
        img[:, :, j] -= mean[j]
    for j in range(3):
        img[:, :, j] /= std[j]

    img = cv2.resize(img, (224, 224))

    h, w = img.shape[:2]
    img = img.astype(np.float32)
    img /= 255
    img = img.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
    img_variable = torch.autograd.Variable(img_tensor, volatile=True)
    return img_variable, h, w


checkpoint = torch.load(PATH)
# print(checkpoint['epoch'])
model = net.EESPNet_Seg(3, s=0.5)
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
# model.cuda()
model.load_state_dict(checkpoint, strict=False)
model.eval()


for f in glob.glob(os.path.join("dataset/data/*.png")):
    image = cv2.imread(f)
    start = time()
    input_img, H_ori, W_ori = image_feed(image)
    input_img.cuda()
    img_out = model(input_img)

    pallete = [
        [0, 0, 0],
        [0, 0, 255],
        [255, 255, 255]
    ]

    classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()

    classMap_numpy_color = np.zeros(
        (224, 224, 3), dtype=np.uint8)

    for idx in range(len(pallete)):
        [r, g, b] = pallete[idx]
        classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
    
    print(time() - start)

    classMap_numpy_color = cv2.resize(classMap_numpy_color, (320, 120))
    cv2.imshow("color", classMap_numpy_color)
    cv2.imshow("image", image)
    cv2.waitKey(0)

#python .\main.py --data_dir dataset/ --inWidth 224 --inHeight 224 --max_epochs 300 --num_workers 1 --batch_size 12 --classes 3 --scaleIn 1 --s 0.5
#python .\gen_cityscapes.py --data_dir dataset/data --inWidth 224 --inHeight 224 --pretrained .\results_espnetv2_0.5\model_best.pth --s 0.5 --classes 3 --overlay True --img_extn png