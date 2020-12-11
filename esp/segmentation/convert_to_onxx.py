import torch
from cnn import SegmentationModel as net
import cv2
import numpy as np
import onnx

PATH = "results_espnetv2_0.5/model_best.pth"

INPUT_WIDTH = 224
INPUT_HEIGHT = 224


def process_input(image):
    mean = [133.49126, 123.00861, 120.21908]
    std = [41.028954, 41.625504, 43.69621]
    img = image.astype(np.float32)
    for j in range(3):
        img[:, :, j] -= mean[j]
    for j in range(3):
        img[:, :, j] /= std[j]
    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    img /= 255
    img = img.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    img_tensor = img_tensor.cuda()
    return img_tensor


ONNX_FILE_PATH = 'model.onnx'

model = net.EESPNet_Seg(3, s=0.5)


model = torch.nn.DataParallel(model)
model = model.cuda()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# model.load_state_dict(torch.load(PATH))
# model.eval()

# image = cv2.imread("dataset/data/frame_1.png")
# input = process_input(image)

# torch.onnx.export(model.module, input,
#                   ONNX_FILE_PATH, export_params=True, opset_version=11)
# torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=[
#                   'input'], output_names=['output'], export_params=True)
onnx_model = onnx.load(ONNX_FILE_PATH)
onnx.checker.check_model(onnx_model)
print(onnx_model)
