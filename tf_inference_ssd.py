#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time

import cv2
import numpy as np

import tensorflow as tf

MODEL_DIR = "all_models"


_COCO_CLASSES_AND_COLORS = [
    ["M", [255, 0, 0]],
    ["F", [0, 255, 0]],
    ["N", [0, 0, 255]],
    ["S", [0, 255, 0]],
    ["MASK", [255, 255, 255]],
    ["LF", [255, 255, 0]],
    ["RF", [255, 0, 255]],
    ["BF", [128, 0, 128]],
    ["OF", [128, 255, 128]],
    ["ROT", [125, 0, 102]],
    ["CI", [0, 0, 255]],
    ["OI", [0, 255, 0]],
    ["EB", [255, 255, 255]],
    ["OM", [128, 128, 128]],
    ["CM", [0, 0, 255]],
    ["EH", [0, 0, 255]],
    ["UP", [0, 120, 0]],
    ["UT", [0, 120, 0]],
    ["HOB", [0, 0, 255]],
    ["MOB", [0, 0, 255]],
    ["BK", [0, 0, 255]],
    ["Id", [0, 255, 0]],
    ["SG", [5, 35, 55]],
    ["TL", [200, 80, 100]],
    ["GOG", [5, 35, 55]]
]
COCO_CLASSES = [d[0] for d in _COCO_CLASSES_AND_COLORS]

# category_index = {
#     i+1:{"id":i+1, "name":"-"} for i in range(91)
# }
category_index = {
    i+1:{"id":i+1, "name":COCO_CLASSES[i]} for i in range(len(COCO_CLASSES))
}

def make_parser():
    parser = argparse.ArgumentParser("TF inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=f"{MODEL_DIR}/ssd",
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


def preprocess_image(img, input_size, swap=(2, 0, 1)):
    
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 0
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 0

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    min_score_thresh=0.0039
):
  
  total_preds = boxes.shape[0]

  print(scores)

  print("num dets", sum( 1 for i in range(total_preds) if scores[i] > min_score_thresh))
  
  for i in range(total_preds):
    
    if scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      ymin, xmin, ymax, xmax = box
    
      ymin, ymax = int(640*ymin), int(640*ymax)
      xmin, xmax = int(640*xmin), int(640*xmax)

    #   print((ymin, xmin, ymax, xmax), classes[i], scores[i])

      class_name = category_index[classes[i]]['name']
      display_str = f"{class_name} {round(100*scores[i])}"

      cv2.rectangle(image, (xmin, ymin), (xmax, ymax), [255, 0, 0], 2)

if __name__ == '__main__':

    args = make_parser().parse_args()

    input_shape = (args.tsize, args.tsize)
    cap = cv2.VideoCapture(args.camid)

    cv2.namedWindow("disp", cv2.WINDOW_NORMAL)

    loaded = tf.saved_model.load(args.model)
    print(list(loaded.signatures.keys()))
    infer = loaded.signatures["serving_default"]
    
    output_nodes = str(list(infer.structured_outputs.keys()))
    print(output_nodes)

    while 1:
        
        ret_val, origin_img = cap.read()

        origin_img = cv2.imread("test.jpg")

        if ret_val:

            # origin_img = origin_img[:, :, ::-1]

            ## needs BGR image
            st = time.time()
            img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
            img, ratio = preprocess_image(img, input_shape)

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(img)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis,...]
            
            # Run inference
            output_dict = infer(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            
            num_detections = int(output_dict.pop('num_detections'))
            output_dict = {key:value[0, :num_detections].numpy() 
                            for key,value in output_dict.items()}
            output_dict['num_detections'] = num_detections

            # detection_classes should be ints.
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int8)
            # output_dict['detection_boxes'] = output_dict['detection_boxes'].astype(np.int8)
            print(time.time()-st, num_detections)
            visualize_boxes_and_labels_on_image_array(
                origin_img,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index
            )
            cv2.imshow("disp", origin_img)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

# python tf_inference_ssd.py --tsize 640 --conf 0.50 --nms 0.6 -m all_models/ssd


"""
0.0 0.9307191967964172 [199.68807983   3.07324219 641.4888916  640.45715332]
3.0 0.9242492318153381 [329.71353149  69.38072205 514.51000977 421.49102783]
11.0 0.7511335611343384 [454.77636719 190.03930664 496.12512207 216.11468506]
11.0 0.749090313911438 [368.10009766 184.02078247 414.54962158 210.43624878]
12.0 0.6687908172607422 [386.47424316 188.86701965 403.8548584  205.36802673]
12.0 0.6091294884681702 [470.22918701 196.56863403 486.08270264 210.37017822]
13.0 0.8710731863975525 [379.76428223 313.3012085  467.97033691 373.49578857]

"""