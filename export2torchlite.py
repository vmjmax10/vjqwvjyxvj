"""
    ALL these code must be running from >custom_yolox >>
"""
import argparse
import os
from loguru import logger

import torch
from torch import nn
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

MODEL_DIR = "all_models"


def make_parser():
    parser = argparse.ArgumentParser("YOLOX torchlite")
    parser.add_argument(
        "--output_lite_path", 
        type=str, 
        default=f"{MODEL_DIR}/yolox_s_vjs.ptl", 
        help="output path of torchlite model"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of torchlite model"
    )
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

    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(args.output_lite_path)

    traced_script_module = traced_script_module.eval()
    traced_script_module = traced_script_module.to(torch.device('cpu'))

    ## 
    with torch.no_grad():
        out_script = traced_script_module(dummy_input)
        print("input shape", dummy_input.shape)
        print("output shape", out_script.shape)

    print("\n\n\n*****************FINISHED******************\n\n")

if __name__ == "__main__":
    main()

    # import torch
    # import io
    # dummy_input = torch.randn(1, 3, 640, 640)
    # traced_script_module = torch.jit.load("all_models/yolox_nano_vjs.ptl")
    # with torch.no_grad():
    #     out_script = traced_script_module(dummy_input)
    #     print(out_script.shape)
    #     np_arr = out_script.cpu().detach().numpy()
    #     print(np_arr.shape, np_arr[0].shape)
    


## YOLOX_NANO
# python export2torchlite.py -f exps/yolox_nano_vjs.py -c all_models/yolox_nano_vjs.pth --output_lite_path all_models/yolox_nano_vjs.ptl --t_size 640
