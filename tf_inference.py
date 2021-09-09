#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
from demo import MODEL_DIR
import os
import time

import cv2
import numpy as np

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis

import tensorflow as tf

MODEL_DIR = "all_models"

def make_parser():
    parser = argparse.ArgumentParser("TF inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=f"{MODEL_DIR}/yolox_s_vjs_tf",
        help="Input your tf saved model path",
    )
    parser.add_argument(
        "--tsize", 
        default=640, 
        type=int, 
        help="test img size, must be fixed",
    )
    parser.add_argument("--conf", default=0.50, type=float, help="test conf")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
 
    return parser


if __name__ == '__main__':

    args = make_parser().parse_args()

    input_shape = (args.tsize, args.tsize)
    cap = cv2.VideoCapture(args.camid)

    cv2.namedWindow("disp", cv2.WINDOW_NORMAL)

    loaded = tf.saved_model.load(args.model)
    print(list(loaded.signatures.keys()))
    infer = loaded.signatures["serving_default"]
    print(list(infer.structured_outputs.keys())[0])
    output_node = str(list(infer.structured_outputs.keys())[0])


    while 1:
        
        ret_val, origin_img = cap.read()

        # origin_img = cv2.imread("test.jpg")

        if ret_val:

            # origin_img = origin_img[:, :, ::-1]
  
            img, ratio = preprocess(origin_img, input_shape)

            img_t = img[None, :, :, :]
 
            pred = infer(tf.convert_to_tensor(img_t, dtype=tf.float32))[output_node].numpy()

            predictions = demo_postprocess(pred, input_shape)[0]

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

# python tf_inference.py --tsize 640 --conf 0.50 --nms 0.6 -m all_models/yolox_s_vjs_tf
# python tf_inference.py --tsize 640 --conf 0.50 --nms 0.6 -m all_models/yolox_nano_vjs_tf

"""
0.0 0.9307191967964172 [199.68807983   3.07324219 641.4888916  640.45715332]
3.0 0.9242492318153381 [329.71353149  69.38072205 514.51000977 421.49102783]
11.0 0.7511335611343384 [454.77636719 190.03930664 496.12512207 216.11468506]
11.0 0.749090313911438 [368.10009766 184.02078247 414.54962158 210.43624878]
12.0 0.6687908172607422 [386.47424316 188.86701965 403.8548584  205.36802673]
12.0 0.6091294884681702 [470.22918701 196.56863403 486.08270264 210.37017822]
13.0 0.8710731863975525 [379.76428223 313.3012085  467.97033691 373.49578857]

"""