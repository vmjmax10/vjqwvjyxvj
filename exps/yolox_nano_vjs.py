#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp
from yolox.models.network_blocks import SiLU, QConv2d, QBatchNorm2d

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 25
        self.depth = 0.33
        self.width = 0.25

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        self.multiscale_range = 2
        # self.random_size = (14, 26)
        self.data_dir = "/content/vjqwvjyxvj/gdrive/MyDrive/DATASET/PRE_VJS"
        self.train_ann = "train_coco.json"
        self.val_ann = "val_coco.json"
        self.train_img_dir = "train"
        self.val_img_dir = "val"
        self.output_dir = "/content/vjqwvjyxvj/gdrive/MyDrive/TRAINED_MODELS/PRE_VJS"
        
        # --------------- transform config ----------------- #
        self.mosaic_prob = 2.0
        self.mixup_prob = 2.0
        self.hsv_prob = 2.0
        self.flip_prob = -1.0
        self.degrees = 5.0
        self.translate = 0.0
        self.mosaic_scale = (0.1, 3.5)
        self.mixup_scale = (0.5, 2.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = False

        # --------------  training config --------------------- #
        self.warmup_epochs = 20
        self.max_epoch = 200
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 48.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 200
        self.min_lr_ratio = 0.05
        self.ema = True
        self.max_labels_tt = 200
        self.max_labels_mosaicd = 200
        self.flip_image = False

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 400
        
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # self.exp_name = "yolox_s_vjs_fp16"

        # -----------------  testing config ------------------ #
        self.test_size = self.input_size
        self.test_conf = 0.01
        self.nmsthre = 0.10


    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, QBatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, 
                self.width, 
                in_channels=in_channels, 
                depthwise=True
            )
            head = YOLOXHead(
                self.num_classes, 
                self.width, 
                in_channels=in_channels, 
                depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model
