#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # ---------------- model config ---------------- #
        self.num_classes = 25
        self.depth = 0.33
        self.width = 0.50

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        self.multiscale_range = 0
        # self.random_size = (14, 26)
        self.data_dir = "/content/vjqwvjyxvj/gdrive/MyDrive/DATASET/VIDEO_ANALYSIS_JS/pre_final"
        self.train_ann = "annotations/instances_train2017.json"
        self.val_ann = "annotations/instances_val2017.json"
        self.train_img_dir = "train2017"
        self.val_img_dir = "val2017"
        self.output_dir = "/content/vjqwvjyxvj/gdrive/MyDrive/TRAINED_MODELS/VIDEO_ANALYSIS_JS"
        
        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = -1.0
        self.degrees = 5.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 30
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 32.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.max_labels_tt = 100
        self.max_labels_mosaicd = 150
        self.flip_image = False

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 100
        
        self.eval_interval = 15
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65
