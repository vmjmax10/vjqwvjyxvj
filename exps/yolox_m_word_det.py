#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # ---------------- model config ---------------- #
        self.num_classes = 4
        self.depth = 0.67
        self.width = 0.75

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (2048, 2048)  # (height, width)
        self.multiscale_range = 0
        # self.random_size = (14, 26)
        self.data_dir = "/content/vjqwvjyxvj/gdrive/MyDrive/DATASET/LAYOUT_WORD_DS/YOLOX"
        self.train_ann = "train_coco.json"
        self.val_ann = "val_coco.json"
        self.train_img_dir = "train"
        self.val_img_dir = "val"
        self.output_dir = "/content/vjqwvjyxvj/gdrive/MyDrive/TRAINED_MODELS/LAYOUT_WORD_DS"
        
        # --------------- transform config ----------------- #
        self.mosaic_prob = 3.0
        self.mixup_prob = 3.0
        self.hsv_prob = 2.0
        self.flip_prob = -1.0
        self.degrees = 5.0
        self.translate = 0.0
        self.mosaic_scale = (0.1, 3.5)
        self.mixup_scale = (0.5, 2.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 600
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.001 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 50
        self.min_lr_ratio = 0.05
        self.ema = True
        self.max_labels_tt = 1000
        self.max_labels_mosaicd = 1500
        self.flip_image = False

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 120
        
        self.eval_interval = 3
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # self.exp_name = "yolox_s_vjs_fp16"

        # -----------------  testing config ------------------ #
        self.test_size = self.input_size
        self.test_conf = 0.01
        self.nmsthre = 0.20
