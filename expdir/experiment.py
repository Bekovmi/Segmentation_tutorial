import json
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from catalyst.data.augmentor import Augmentor
from catalyst.data.dataset import ListDataset
from catalyst.data.reader import ScalarReader, ReaderCompose, ImageReader
from catalyst.data.sampler import BalanceClassSampler
from catalyst.dl import ConfigExperiment

from segmentationDataset import SegmentationDataset
from transforms import pre_transforms, post_transforms, hard_transform, Compose
class Experiment(ConfigExperiment):
    @staticmethod
    def get_transforms(
        stage: str = None,
        mode: str = None,
        image_size: int = 224,
        one_hot_classes: int = None
    ):
        if mode == "train":
            post_transform_fn = Compose(
                [pre_transforms(image_size=image_size), hard_transform(image_size=image_size),
                 post_transforms()])
        elif mode in ["valid", "infer"]:
            post_transform_fn = Compose([pre_transforms(image_size=image_size), post_transforms()])
        else:
            raise NotImplementedError()
        return post_transform_fn

    def get_datasets(
        self,
        stage: str,
        datapath: str = None,
        in_csv: str = None,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        image_size: int = 224
    ):
        datasets = collections.OrderedDict()
        df_train = pd.read_csv(in_csv_train)
        df_valid = pd.read_csv(in_csv_valid)
        for source, mode in zip(
            (df_train, df_valid), ("train", "valid")):
            datasets[mode] = SegmentationDataset(source,datapath,augmentation = self.get_transforms(stage=stage,mode=mode,image_size=image_size))
        return datasets
    def _postprocess_mode_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.Module
        return model_
