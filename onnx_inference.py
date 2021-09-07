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

    while 1:
        
        ret_val, origin_img = cap.read()

        # origin_img = cv2.imread("./js_inference/test.jpg")

        if ret_val:

            # origin_img = origin_img[:, :, ::-1]
  
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

# python onnx_inference.py --tsize 640 --conf 0.50 --nms 0.6 -m all_models/yolox_s_vjs.onnx

"""
0.0    0.8796307444572449   [151.20025635   5.99160767 630.61602783 640.97021484]
3.0    0.9059643745422363   [329.68307495  86.60913086 511.31381226 420.68209839]
11.0   0.7884166836738586   [361.39559937 179.9336853  417.67819214 216.37127686]
11.0   0.764872133731842    [451.46508789 187.32156372 499.07800293 220.98202515]
12.0   0.7573755979537964   [388.18676758 189.01637268 405.98254395 206.76609802]
12.0   0.6793655157089233   [470.34509277 196.39561462 489.14025879 212.06889343]
13.0   0.8700669407844543   [369.07711792 307.98425293 474.65847778 380.06402588]


"""