import onnx
# import warnings
from onnx_tf.backend import prepare

model = onnx.load("models/yolox_nano_vjs.onnx")
print(onnx.helper.printable_graph(model.graph))

tf_rep = prepare(model, logging_level="DEBUG", gen_tensor_dict=False) 

print(tf_rep.inputs) # Input nodes to the model
print("-----")
print(tf_rep.outputs) # Output nodes from the model
print("-----")
print(tf_rep.tensor_dict) # All nodes in the model

tf_rep.export_graph("models/yolox_nano_vjs_tf")


if 1:
    import tensorflow as tf
    loaded = tf.saved_model.load("models/yolox_nano_vjs_tf")
    print(list(loaded.signatures.keys()))
    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)

# python tools/onnx2tf.py