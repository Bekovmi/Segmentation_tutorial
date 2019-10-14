from typing import List, Dict
import random
import numpy as np
import cv2

import albumentations as albu
from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, LongestMaxSize, PadIfNeeded,
    Normalize, HueSaturationValue, ShiftScaleRotate, RandomGamma,
    IAAPerspective, JpegCompression, ToGray, ChannelShuffle, RGBShift, CLAHE,
    RandomBrightnessContrast, RandomSunFlare, Cutout, OneOf
)
from albumentations.pytorch import ToTensor

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def pre_transforms(image_size=224):
    return Compose(
        [
            LongestMaxSize(max_size=image_size),
            PadIfNeeded(
                image_size, image_size, border_mode=cv2.BORDER_CONSTANT
            ),
        ]
    )

def post_transforms():
    return Compose([Normalize(), ToTensor()])

def hard_transform(image_size=224, p=0.5):
    transforms = [
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=p
        ),]
    transforms = Compose(transforms)
    return transforms
