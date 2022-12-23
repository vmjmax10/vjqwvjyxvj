import tensorrt as trt
import numpy as np, cv2
import os
import time

import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

                
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size

            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self,x:np.ndarray, batch_size=1):
        
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host, x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size, -1) for out in self.outputs]

if __name__ == "__main__":
 
    # trt_engine_path = os.path.join("..","models","main.trt")

    infer_size = 2048
    swap=(2, 0, 1)
    num_samples = 15
    img = cv2.imread("test.jpg")
    img = np.ones((infer_size, infer_size, 3), dtype=np.uint8)*255
    sample_images = [
        img
        # np.ones((infer_size,infer_size, 3), dtype=np.uint8)
        # np.random.randint(0, 256, size=(infer_size,infer_size, 3), dtype=np.uint8)
        for _ in range(num_samples)
    ]

    img_batch = []
    for idx in range(num_samples):
        resized_img = img.transpose(swap)
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)
        img_s = resized_img[None, :, :, :]
        img_batch.append(img_s)

    trt_engine_path = "all_models/yolox_s_layout_word_det_v3_sign.trt"
    model = TrtModel(trt_engine_path, max_batch_size=1)

    for idx in range(num_samples):
        st = time.time()
        result = model(img_batch[idx], batch_size=1)
        tt = time.time()-st
        print("Done", tt, result[0].shape)

    # trt_engine_path1 = "all_models/yolox_nano_vjs_dyn.trt"
    # model1 = TrtModel(trt_engine_path1, max_batch_size=num_samples)

    # for _ in range(4):
    #     st = time.time()
    #     result = model1(np.concatenate(img_batch[:2]), batch_size=2)

    #     print(len(result))
    #     tt = time.time()-st
    #     print("BATCH", tt)