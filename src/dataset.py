from torch.utils.data import Dataset
from skimage.io import imread as gif_imread
import cv2
from catalyst import utils
class SegmentationDataset(Dataset):
    def __init__(self, dataframe,augmentation=None):
        self.dataframe = dataframe
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.dataframe.iloc[idx,0]
        image = cv2.imread(image_path)
        
        result = {"image": image}
        
        mask = gif_imread(self.dataframe.iloc[idx,1])
        result["mask"] = mask
        
        if self.transforms is not None:
            result = self.transforms(**result)
        
        result["filename"] = image_path.name

        return result
