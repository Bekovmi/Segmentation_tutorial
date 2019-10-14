from typing import Callable, Dict, List
import numpy as np
import imageio

from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, list_data: List[Dict], dict_transform: Callable = None):
        self.data = list_data
        self.dict_transform = dict_transform

    def __getitem__(self, index: int) -> Dict:
        dict_ = self.data[index]
        dict_ = {
            "image": np.asarray(imageio.imread(dict_["image"], pilmode="RGB")),
            "mask": np.asarray(imageio.imread(dict_["mask"])),
        }
        if self.dict_transform is not None:
            dict_ = self.dict_transform(**dict_)

        return dict_

    def __len__(self) -> int:
        return len(self.data)
