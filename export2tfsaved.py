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

MODEL_DIR = "all_models"


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output_onnx_path", 
        type=str, 
        default=f"{MODEL_DIR}/yolox_s_vjs.onnx", 
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
    parser.add_argument("-c", "--ckpt", default=f"{MODEL_DIR}/yolox_s_vjs.pth", type=str, help="ckpt path")
    parser.add_argument(
        "-size", 
        "--t_size", 
        default=640, 
        type=int, 
        help="model input square size, if provided, test size willbe replaced"
    )
    
    parser.add_argument("--tf_saved", action="store_true", help="convert into tf saved model or not")
    parser.add_argument(
        "--output_tf_saved_path", 
        type=str, 
        default=f"{MODEL_DIR}/yolox_s_vjs_tf", 
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

    dynamic_axes_dict = {
        "image_batch": [
            0, 
            2, 
            3
        ], 
        "output": [
            0, 
            1, 
            2
        ]
    }

    torch.onnx._export(
        model,
        dummy_input,
        args.output_onnx_path,
        input_names=["image_batch"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes_dict
    )
    logger.info("generated onnx model named {}".format(args.output_onnx_path))

    # if not args.no_onnxsim:
    #     import onnx, onnxruntime

    #     from onnxsim import simplify

    #     # use onnxsimplify to reduce reduent model.
    #     onnx_model = onnx.load(args.output_onnx_path)
    #     model_simp, check = simplify(
    #         onnx_model,
    #         dynamic_input_shape=True,
    #     )
    #     assert check, "Simplified ONNX model could not be validated"
    #     onnx.save(model_simp, args.output_onnx_path)
    #     logger.info("generated simplified onnx model named {}".format(args.output_onnx_path))

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

        if not os.path.exists(args.output_tf_saved_path):
            os.mkdir(args.output_tf_saved_path)

        tf_rep.export_graph(args.output_tf_saved_path)
        print("Converted model into tensorflow saved model", args.output_tf_saved_path)
        print("\n\n\n*****************DONE******************\n\n")

        print("\nCross checking tf model")
        import tensorflow as tf
        loaded = tf.saved_model.load(args.output_tf_saved_path)
        print(list(loaded.signatures.keys()))
        infer = loaded.signatures["serving_default"]
        print(infer.structured_outputs)

    print("\n\n\n*****************FINISHED******************\n\n")


if __name__ == "__main__":
    main()

    ## 

## ONNX ONLY
# python export2tfsaved.py -f exps/yolox_nano_vjs.py -c all_models/yolox_nano_vjs.pth --output_onnx_path all_models/yolox_nano_vjs_dyn.onnx --t_size 640



## YOLOX_NANO
# python export2tfsaved.py  --tf_saved -f exps/yolox_nano_vjs.py -c all_models/yolox_nano_vjs.pth --output_onnx_path all_models/yolox_nano_vjs.onnx --output_tf_saved_path all_models/yolox_nano_vjs_tf --t_size 640


## YOLOX_S
# python export2tfsaved.py --tf_saved --t_size 640


## ZIM ID
# python export2tfsaved.py -f exps/yolox_s_zim_4_ids.py -c all_models/yolox_s_zim_4_ids.pth --output_onnx_path all_models/yolox_s_zim_4_ids.onnx --t_size 640

## WORD DET
# python export2tfsaved.py -f exps/yolox_s_word_det.py -c all_models/yolox_s_word_det.pth --output_onnx_path all_models/yolox_s_word_det.onnx --t_size 1280


## WORD DET M
# python export2tfsaved.py -f exps/yolox_m_word_det.py -c all_models/yolox_m_word_det.pth --output_onnx_path all_models/yolox_m_word_det.onnx --t_size 1536


## LAYOUT M
# python export2tfsaved.py -f exps/yolox_m_layout.py -c all_models/yolox_m_layout.pth --output_onnx_path all_models/yolox_m_layout.onnx --t_size 768


## LAYOUT S
# python export2tfsaved.py -f exps/yolox_s_layout.py -c all_models/yolox_s_layout.pth --output_onnx_path all_models/yolox_s_layout.onnx --t_size 768


## LAYOUT & WORD det - M
# python export2tfsaved.py -f exps/yolox_m_word_det.py -c all_models/yolox_m_lay_word_det.pth --output_onnx_path all_models/yolox_m_lay_word_det.onnx --t_size 2048


## LAYOUT & WORD det - S
# python export2tfsaved.py -f exps/yolox_s_word_det.py -c all_models/yolox_s_lay_word_det.pth --output_onnx_path all_models/yolox_s_lay_word_det.onnx --t_size 2048

# python export2tfsaved.py -f exps/yolox_s_lwdet.py -c all_models/yolox_s_lwdet.pth --output_onnx_path all_models/yolox_s_lwdet.onnx --t_size 2048
