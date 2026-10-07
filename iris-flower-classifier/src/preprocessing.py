"""Load, clean and split the Iris dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (FEATURES, PROCESSED_DATA, RANDOM_STATE, RAW_DATA,
                     TARGET, TEST_SIZE)


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the Id column and duplicate rows."""
    df = df.copy()
    if "Id" in df.columns:
        df = df.drop(columns="Id")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def get_clean_data(save: bool = True) -> pd.DataFrame:
    df = clean(load_raw())
    if save:
        PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA, index=False)
    return df


def split(df: pd.DataFrame):
    """Stratified train/test split. Labels stay as species names."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)
