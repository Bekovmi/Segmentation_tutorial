import cv2
from albumentations import (
    Compose, LongestMaxSize, PadIfNeeded, Normalize, ShiftScaleRotate,
    RandomGamma
)
from albumentations.pytorch import ToTensor
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def pre_transforms(image_size=256):
    return Compose([
        LongestMaxSize(max_size=image_size),
        PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
    ])

def post_transforms():
    return Compose([Normalize(), ToTensor()])

def hard_transform(image_size=256):
    return Compose([
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
    ])