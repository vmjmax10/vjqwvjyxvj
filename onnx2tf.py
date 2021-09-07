import onnx
import os
# import warnings
from onnx_tf.backend import prepare

model_name = "yolox_s"

model = onnx.load(f"all_models/{model_name}_vjs.onnx")
print(onnx.helper.printable_graph(model.graph))

tf_rep = prepare(model, logging_level="DEBUG", gen_tensor_dict=False) 

print(tf_rep.inputs) # Input nodes to the model
print("-----")
print(tf_rep.outputs) # Output nodes from the model
print("-----")
print(tf_rep.tensor_dict) # All nodes in the model


output_path = f"all_models/{model_name}_vjs_tf"

if not os.path.exists(output_path):
    os.mkdir(output_path)

tf_rep.export_graph(output_path)


if 1:
    import tensorflow as tf
    loaded = tf.saved_model.load(f"all_models/{model_name}_vjs_tf")
    print(list(loaded.signatures.keys()))
    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)

# python onnx2tf.py