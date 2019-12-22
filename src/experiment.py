import collections
import pandas as pd
from catalyst.dl import ConfigExperiment
from .dataset import SegmentationDataset
from .transforms import (
    pre_transforms, post_transforms, hard_transform, Compose
)

class Experiment(ConfigExperiment):
    @staticmethod
    def get_transforms(
        stage: str = None, mode: str = None, image_size: int = 256,
    ):
        pre_transform_fn = pre_transforms(image_size=image_size)
        if mode == "train":
            post_transform_fn = Compose([
                hard_transform(image_size=image_size),
                post_transforms()
            ])
        elif mode in ["valid", "infer"]:
            post_transform_fn = post_transforms()
        else:
            raise NotImplementedError()
        
        transform_fn = Compose([pre_transform_fn, post_transform_fn])
        return transform_fn
    def get_datasets(
        self,
        stage: str,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        image_size: int = 256,
    ):
        datasets = collections.OrderedDict()
        for source, mode in zip(
            (in_csv_train, in_csv_valid), ("train", "valid")
        ):
            dataframe = pd.read_csv(source)
            datasets[mode] = SegmentationDataset(
                dataframe.to_dict('records'),
                dict_transform=self.get_transforms(
                    stage=stage, mode=mode, image_size=image_size
                ),
            )
        return datasets