from skimage.io import imread as gif_imread

from torch.utils.data import Dataset

from catalyst import utils


class SegmentationDataset(Dataset):
    def __init__(self, dataframe, path_img, augmentation=None):
        self.dataframe = dataframe
        self.augmentation = augmentation
        self.path_img = path_img

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.dataframe.iloc[0,idx]
        image = utils.imread(image_path)
        
        result = {"image": image}
        
        if self.masks is not None:
          mask = gif_imread(self.masks[1,idx])
          result["mask"] = mask
        
        if self.transforms is not None:
            result = self.transforms(**result)
        
        result["filename"] = image_path.name

        return result
