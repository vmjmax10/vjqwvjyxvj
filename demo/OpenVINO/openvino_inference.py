#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import logging as log
import os
import sys

import cv2
import numpy as np

from openvino.inference_engine import IECore

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    parser.add_argument("--conf", default=0.65, type=float, help="test conf")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    args.add_argument(
        '-m',
        '--model',
        required=True,
        type=str,
        help='Required. Path to an .xml or .onnx file with a trained model.')

    args.add_argument(
        '-d',
        '--device',
        default='CPU',
        type=str,
        help='Optional. Specify the target device to infer on; CPU, GPU, \
              MYRIAD, HDDL or HETERO: is acceptable. The sample will look \
              for a suitable plugin for device specified. Default value \
              is CPU.')

    return parser.parse_args()


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    log.info(f'Reading the network: {args.model}')
    # (.xml and .bin files) or (.onnx file)
    net = ie.read_network(model=args.model)

    if len(net.input_info) != 1:
        log.error('Sample supports only single input topologies')
        return -1
    if len(net.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Set input and output precision manually
    net.input_info[input_blob].precision = 'FP32'
    net.outputs[out_blob].precision = 'FP16'

    # Get a number of classes recognized by a model
    num_of_classes = max(net.outputs[out_blob].shape)
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device)

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("disp", cv2.WINDOW_NORMAL)

    while 1:

        ret_val, origin_img = cap.read()
        origin_img = cv2.imread("./js_inference/test.jpg")
    
        if ret_val:
  
            origin_img = cv2.imread(args.input)
            _, _, h, w = net.input_info[input_blob].input_data.shape
            img, ratio = preprocess(origin_img, (h, w))
                
            res = exec_net.infer(inputs={input_blob: img})
            res = res[out_blob]

            predictions = demo_postprocess(res, (h, w), p6=False)[0]
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
                for b, c, cs in zip(final_boxes, final_scores, final_cls_inds):
                    print(cs, c, b)  
                exit()

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
    
    if __name__ == '__main__':
        sys.exit(main())
