import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
def prepared_data(foldername):
    ROOT = Path(f"{foldername}/")
    train_image_path = ROOT / "train"
    train_mask_path = ROOT / "train_masks"
    test_image_path = ROOT / "test"
    ALL_IMAGES = sorted(train_image_path.glob("*.jpg"))
    ALL_MASKS = sorted(train_mask_path.glob("*.gif"))
    data = pd.DataFrame({"image":ALL_IMAGES, "mask":ALL_MASKS})
    X_train, X_test = train_test_split(data, test_size=0.1, random_state=42,shuffle=True)
    X_train.to_csv(f"{foldername}/dataset_train.csv",index=False)
    X_test.to_csv(f"{foldername}/dataset_valid.csv",index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='file')
    parser.add_argument('foldername', type=str, help=' foldername ')

    args = parser.parse_args()
    prepared_data(args.foldername)

