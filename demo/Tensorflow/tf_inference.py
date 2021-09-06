#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time

import cv2
import numpy as np

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis, _pred

import tensorflow as tf

def make_parser():
    parser = argparse.ArgumentParser("TF inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="models/yolox_s_vjs_tf",
        help="Input your tf saved model path",
    )
    parser.add_argument(
        "--tsize", 
        default=None, 
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
    print(infer.structured_outputs)

    while 1:
        
        ret_val, origin_img = cap.read()

        # origin_img = cv2.imread("./js_inference/test.jpg")

        if ret_val:

            # origin_img = origin_img[:, :, ::-1]
  
            img, ratio = preprocess(origin_img, input_shape)

            img_t = img[None, :, :, :]
 
            pred = infer(tf.convert_to_tensor(img_t, dtype=tf.float32))["output"].numpy()

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

# python demo/Tensorflow/tf_inference.py --tsize 640 --conf 0.50 --nms 0.6 -m models/yolox_s_vjs_tf
# python demo/Tensorflow/tf_inference.py --tsize 640 --conf 0.50 --nms 0.6 -m models/yolox_nano_vjs_tf

