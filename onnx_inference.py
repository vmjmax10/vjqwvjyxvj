#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time

import cv2
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis

MODEL_DIR = "all_models"


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=f"{MODEL_DIR}/yolox_s_vjs.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "--tsize", 
        default=None, 
        type=int, 
        help="test img size",
    )
    parser.add_argument("--conf", default=0.50, type=float, help="test conf")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


if __name__ == '__main__':

    args = make_parser().parse_args()

    input_shape = (args.tsize, args.tsize)
    cap = cv2.VideoCapture(args.camid)

    cv2.namedWindow("disp", cv2.WINDOW_NORMAL)

    session = onnxruntime.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name

    # import torch
    # import io
    # traced_script_module = torch.jit.load("all_models/yolox_nano_vjs.ptl")

    while 1:
        
        ret_val, origin_img = cap.read()

        origin_img = cv2.imread("test.jpg")
        # origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

        if ret_val:

            # origin_img = origin_img[:, :, ::-1]
  
            img, ratio = preprocess(origin_img, input_shape)

            img_t = img[None, :, :, :]
            
            # print(img_t.flatten()[:30])

            ort_inputs = {input_name: img_t}
            output = session.run(None, ort_inputs)
            pred = output[0]

            # with torch.no_grad():
            #     out_script = traced_script_module(torch.from_numpy(img_t))
            #     pred = out_script.cpu().detach().numpy()
            
            predictions = demo_postprocess(pred, input_shape, p6=args.with_p6)[0]

            # print("\n\n", predictions.flatten()[:30])

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= ratio

            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=args.nms, score_thr=args.conf)
            
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                # for b, c, cs in zip(final_boxes, final_scores, final_cls_inds):
                #     print(cs, c, b)  
                # exit()

                origin_img = vis(
                    origin_img, 
                    final_boxes, 
                    final_scores, 
                    final_cls_inds,
                    conf=args.conf, 
                    class_names=COCO_CLASSES
                )

            cv2.imshow("disp", origin_img)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
    

    if 0:

        import cv2
        import json

        with open("folder_wise.json") as f:
            f_wise = json.load(f)

        main_dir = "3e83ee19a84517d72ef7f4f53d8b92df"
        cv2.namedWindow("disp", cv2.WINDOW_NORMAL)

        for folder, info in f_wise.items():

            if folder != "e2b5f95453d95db8b57fe2f05a2f1114":
                continue
            
            for img_info in info:

                origin_img = cv2.imread(img_info["image_path"])
                if origin_img is None:
                    continue

                if "two" not in img_info["report"]:
                    continue

                img, ratio = preprocess(origin_img, input_shape)

                img_t = img[None, :, :, :]
                ort_inputs = {input_name: img_t}
                output = session.run(None, ort_inputs)
                pred = output[0] 
            
                predictions = demo_postprocess(pred, input_shape, p6=args.with_p6)[0]

                boxes = predictions[:, :4]
                scores = predictions[:, 4:5] * predictions[:, 5:]

                boxes_xyxy = np.ones_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
                boxes_xyxy /= ratio

                dets = multiclass_nms(boxes_xyxy, scores, nms_thr=args.nms, score_thr=args.conf)
                
                if dets is not None:
                    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                   
                    origin_img = vis(
                        origin_img, 
                        final_boxes, 
                        final_scores, 
                        final_cls_inds,
                        conf=args.conf, 
                        class_names=COCO_CLASSES
                    )

                    text = img_info["report"]
                    cv2.putText(origin_img, text, (15, 15), 1, 1, (0, 0, 255), thickness=2)

                    cv2.imshow("disp", origin_img)
                    k = cv2.waitKey(0)

                    if k == ord("q"):
                        exit()




# python onnx_inference.py --tsize 640 --conf 0.50 --nms 0.6 -m all_models/yolox_s_vjs.onnx

"""
0.0    0.8796307444572449   [151.20025635   5.99160767 630.61602783 640.97021484]
3.0    0.9059643745422363   [329.68307495  86.60913086 511.31381226 420.68209839]
11.0   0.7884166836738586   [361.39559937 179.9336853  417.67819214 216.37127686]
11.0   0.764872133731842    [451.46508789 187.32156372 499.07800293 220.98202515]
12.0   0.7573755979537964   [388.18676758 189.01637268 405.98254395 206.76609802]
12.0   0.6793655157089233   [470.34509277 196.39561462 489.14025879 212.06889343]
13.0   0.8700669407844543   [369.07711792 307.98425293 474.65847778 380.06402588]

BGR

ONNX - NANO
0.0     0.907875657081604 [154.65299988  -0.96533203 619.34197998 638.40936279]
3.0     0.8908939957618713 [325.90454102  74.47549438 516.0791626  425.46704102]
11.0    0.7369137406349182 [361.7456665  180.99505615 415.23077393 214.84213257]
11.0    0.7054691314697266 [451.69836426 186.34512329 498.82574463 219.56820679]
12.0    0.6378061771392822 [386.11962891 189.35546875 405.40228271 205.73449707]
12.0    0.5513284802436829 [468.50366211 195.19206238 485.78607178 210.97566223]
13.0    0.8277146816253662 [374.25543213 312.50985718 473.92456055 376.11923218]

TORCHLITE -- SAME

[1.6481564e+01 1.0631575e+01 2.9691843e+01 1.8778624e+01 1.7579066e-04
 1.8655799e-02 1.5943920e-02 6.8818159e-02 1.0833421e-03 2.5652717e-03
 1.4339109e-03 1.0802632e-03 1.6945130e-04 1.4864199e-03 5.4653501e-05
 8.5584968e-03 6.1155930e-02 5.9658574e-04 1.4957828e-03 2.8113712e-04
 2.4144218e-04 5.4032207e-05 2.7975504e-04 2.7519750e-04 3.2068237e-05
 1.8705578e-04 2.4108174e-04 1.1810803e-05 1.0758534e-03 5.9694209e-04]


RGB

TORCHLITE

0.0 0.8946840167045593 [137.97776794  -2.28338623 631.61541748 638.2701416 ]
3.0 0.8526270985603333 [326.53182983  76.33294678 512.94921875 433.47241211]
11.0 0.7436059713363647 [358.6920166  182.04138184 415.23999023 219.11669922]
11.0 0.6794808506965637 [450.6031189  186.81718445 499.39889526 224.93797302]
12.0 0.5860695242881775 [386.03845215 190.71211243 405.87493896 206.73774719]
13.0 0.8298873901367188 [373.19888306 310.52035522 477.20108032 378.06918335]


[1.7181921e+01 1.1877457e+01 3.1555412e+01 2.1454742e+01 1.1494675e-04
 1.2374271e-02 9.5842397e-03 8.6713165e-02 1.3540341e-03 4.7236546e-03
 1.5782214e-03 7.0327905e-04 1.1387052e-04 1.9656415e-03 6.8102272e-05
 4.1363635e-03 7.4744366e-02 5.8566441e-04 7.2419085e-04 8.7244298e-05
 1.0605199e-04 3.2219024e-05 1.8645798e-04 1.6276697e-04 3.9451879e-05
 1.0676206e-04 3.3647145e-04 9.0329368e-06 1.1077146e-03 5.2731851e-04]


"""

