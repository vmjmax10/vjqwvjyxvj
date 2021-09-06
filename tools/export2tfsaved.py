"""
    ALL these code must be running from >custom_yolox >>
"""
import argparse
import os
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

MODEL_DIR = "models"


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output_onnx_path", 
        type=str, 
        default="models/yolox_s_vjs.onnx", 
        help="output path of onnx model"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--no_onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/yolox_s_vjs.py",
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default="models/yolox_s_vjs.pth", type=str, help="ckpt path")
    parser.add_argument(
        "-size", 
        "--t_size", 
        default=0, 
        type=int, 
        help="model input square size, if provided, test size willbe replaced"
    )
    
    parser.add_argument("--tf_saved", action="store_true", help="convert into tf saved model or not")
    parser.add_argument(
        "--output_tf_saved_path", 
        type=str, 
        default="models/yolox_s_vjs_tf", 
        help="output path of tf saved model"
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser

@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    logger.info("loading checkpoint done.")

    if args.t_size != 0:
        dummy_input = torch.randn(1, 3, args.t_size, args.t_size)
    else:
        dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])

    torch.onnx._export(
        model,
        dummy_input,
        args.output_onnx_path,
        input_names=["image_batch"],
        output_names=[""],
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(args.output_onnx_path))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_onnx_path)
        logger.info("generated simplified onnx model named {}".format(args.output_onnx_path))

    if args.tf_saved and os.path.exists(args.output_onnx_path):

        import onnx
        from onnx_tf.backend import prepare

        model = onnx.load(args.output_onnx_path)
        print(onnx.helper.printable_graph(model.graph))

        tf_rep = prepare(model, logging_level="DEBUG", gen_tensor_dict=False) 

        print(tf_rep.inputs) # Input nodes to the model
        print("-----")
        print(tf_rep.outputs) # Output nodes from the model
        print("-----")
        print(tf_rep.tensor_dict) # All nodes in the model

        tf_rep.export_graph(args.output_tf_saved_path)
        print("Converted model into tensorflow saved model", args.output_tf_saved_path)
        print("\n\n\n*****************DONE******************\n\n")

        print("\ncross checking model")
        import tensorflow as tf
        loaded = tf.saved_model.load(args.output_tf_saved_path)
        print(list(loaded.signatures.keys()))
        infer = loaded.signatures["serving_default"]
        print(infer.structured_outputs)

    print("\n\n\n*****************FINISHED******************\n\n")

if __name__ == "__main__":
    main()


## YOLOX_NANO
# python tools/export2tfsaved.py -f exps/yolox_nano_vjs.py -c models/yolox_nano_vjs.pth --output_onnx_path models/yolox_nano_vjs.onnx --output_tf_saved_path models/yolox_nano_vjs_tf --t_size 768


## YOLOX_S
# python tools/export2tfsaved.py --tf_saved --t_size 640
