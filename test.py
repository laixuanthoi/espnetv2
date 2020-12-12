from torch2trt.torch2trt import TRTModule
import torch

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('model_best.pth'))