# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:54
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : config.py
"""

"""
import os
from os.path import join
from enum import Enum


# Paths
ROOT = os.getcwd()
DATASET = join(ROOT, "dataset")
RAW_DATA = join(ROOT, "raw_data")
CLAHE_DATA = join(ROOT, "clahe_data")
NLMEANS_DATA = join(ROOT, "nlmeans_data")
ESRGAN_DATA = join(ROOT, "esrgan_data")


class UNetConfig:

    def __init__(self,
                 epochs = 500,  # Number of epochs
                 batch_size = 1,    # Batch size
                 validation = 10.0,   # Percent of the data that is used as validation (0-100)
                 out_threshold = 0.5,

                 optimizer='Adam',
                 lr = 0.0001,     # learning rate
                 lr_decay_milestones = [20, 50],
                 lr_decay_gamma = 0.9,
                 weight_decay=1e-8,
                 momentum=0.9,
                 nesterov=True,

                 n_channels = 1, # Number of channels in input images
                 n_classes = 8,  # Number of classes in the segmentation
                 scale = 1,    # Downscaling factor of the images

                 load = False,   # Load model from a .pth file
                 save_cp = False,

                 model='NestedUNet',
                 bilinear = True,
                 deepsupervision = False,
                 ):
        super(UNetConfig, self).__init__()

        self.images_dir = join(CLAHE_DATA, "images")
        self.masks_dir = join(CLAHE_DATA, "masks")
        self.checkpoints_dir = join(CLAHE_DATA, "checkpoints")

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold

        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale = scale

        self.load = load
        self.save_cp = save_cp

        self.model = model
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision


