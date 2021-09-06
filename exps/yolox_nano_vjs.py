#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (768, 768)
        self.mosaic_scale = (0.5, 1.5)
        
        self.test_size = (640, 640)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = True

        # Define yourself dataset path
        self.data_dir = "/content/YOLOX/gdrive/MyDrive/DATASET/VIDEO_ANALYSIS_JS/CUSTOM_DATASET"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.output_dir = "/content/YOLOX/gdrive/MyDrive/TRAINED_MODELS/VIDEO_ANALYSIS_JS"

        self.num_classes = 25

        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0.0
        self.basic_lr_per_img = 0.01 / 32.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.data_num_workers = 4
        
        self.multiscale_range = 0
        # self.random_size = (10, 20)

        self.eval_interval = 25
        self.print_interval = 40

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model
