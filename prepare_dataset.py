import pathlib
import pandas as pd
from sklearn import model_selection


def main(
    datapath: str = "./data", valid_size: float = 0.2, random_state: int = 42
):
    datapath = pathlib.Path(datapath)
    dataframe = pd.DataFrame({
        "image": sorted((datapath / "train").glob("*.jpg")),
        "mask": sorted((datapath / "train_masks").glob("*.gif"))
    })

    df_train, df_valid = model_selection.train_test_split(
        dataframe,
        test_size=valid_size,
        random_state=random_state,
        shuffle=False
    )
    for source, mode in zip((df_train, df_valid), ("train", "valid")):
        source.to_csv(datapath / f"dataset_{mode}.csv", index=False)


if __name__ == "__main__":
    main()

