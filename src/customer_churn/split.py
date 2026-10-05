from pathlib import Path

import pandas as pd

from .config import PROCESSED_DATA_PATH, RANDOM_SEED


def split_data() -> tuple[Path, Path, Path]:
    """Split processed dataframe into train, val, test.
    Returns paths to parquet files.
    """
    from sklearn.model_selection import train_test_split

    df: pd.DataFrame = pd.read_parquet(PROCESSED_DATA_PATH)
    # First split 70% train+val, 30% test
    train_val: pd.DataFrame
    test: pd.DataFrame
    train_val, test = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df.loc[:, "Churn"],
    )  # type: ignore[assignment]
    # Split train_val into 70% train, 30% val (relative to train_val)
    train: pd.DataFrame
    val: pd.DataFrame
    train, val = train_test_split(
        train_val,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=train_val.loc[:, "Churn"],
    )  # type: ignore[assignment]
    # Paths
    base = PROCESSED_DATA_PATH.parent
    train_path = base / "train.parquet"
    val_path = base / "val.parquet"
    test_path = base / "test.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)

    # Print distributions
    def dist(df_split: pd.DataFrame, name: str) -> None:
        print(f"{name} shape: {df_split.shape}")
        print(
            df_split.loc[:, "Churn"]
            .value_counts(normalize=True)
            .rename("Proportion")
            .mul(100)
            .round(2)
        )

    dist(train, "Train")
    dist(val, "Validation")
    dist(test, "Test")
    return train_path, val_path, test_path


if __name__ == "__main__":
    split_data()
