import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import onnx
import os
import cv2 

ONNX_FILE_PATH = 'model.onnx'
# onnx_model = onnx.load(ONNX_FILE_PATH)
# onnx.checker.check_model(onnx_model)


INPUT_WIDTH = 224
INPUT_HEIGHT = 224

pallete = [
    [0,0,0],
    [0,0,255],
    [255,255,255]
]

TRT_LOGGER = trt.Logger()

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
    # img_tensor = img_tensor.cuda()
    return img_tensor


def process_output(output_data):
    classMap_numpy = output_data[0].max(0)[1].byte().cpu().data.numpy()
    classMap_numpy_color = np.zeros(
                (INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    for idx in range(len(pallete)):
        [r, g, b] = pallete[idx]
        classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
    return classMap_numpy_color

def get_engine(onnx_file_path='', engine_file_path=""):
    network_flags = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_flags) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 224, 224]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

engine = get_engine(engine_file_path="engine_lane_segment.trt")
context = engine.create_execution_context()
for binding in engine:
    if engine.binding_is_input(binding):  # we expect only one input
        input_shape = engine.get_binding_shape(binding)
        input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
        device_input = cuda.mem_alloc(input_size)
    else:  # and one output
        output_shape = engine.get_binding_shape(binding)
        host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
        device_output = cuda.mem_alloc(host_output.nbytes)

    stream = cuda.Stream()

    image = cv2.imread("frame_1.png")
    host_input = np.array(process_input(image).numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    output_mask = process_output(output_data)
    print(output_mask)
    

