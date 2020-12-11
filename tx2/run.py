import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger()
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)


def build_engine(onnx_file_path):

    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
        print('Completed parsing of ONNX file')


builder.max_workspace_size = 1 << 30
builder.max_batch_size = 1

if builder.platform_has_fast_fp16:
    builder.fp16_mode = True
