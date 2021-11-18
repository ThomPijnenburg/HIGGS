import requests
import torch

import pandas as pd

from pathlib import Path
from tqdm import tqdm

from torch.utils.data import TensorDataset

from typing import Tuple, Union

from higgs.logging import get_logger

logger = get_logger(__name__)


def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm(unit="B", unit_scale=True, position=0,
                    leave=True, total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)
    return filename


def df_to_tensordataset(df: pd.DataFrame, target_col: int = 0):
    feats = torch.from_numpy(df.drop(columns=[target_col]).values).float()
    target = torch.from_numpy(df[0].values)
    tensor_dataset = TensorDataset(feats, target)
    return tensor_dataset


def HiggsDataset(data_dir: Union[str, Path]) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    TODO fix data path
    Download Higss dataset
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    filename = "HIGGS.csv.gz"
    data_dir = Path(data_dir).joinpath("higgs")
    raw_file = data_dir.joinpath(filename)

    train_size, valid_size, test_size = (10000000, 500000, 500000)
    feature_cols = list(range(0, 22))

    if not raw_file.is_file():
        logger.info(f"Downloading dataset from {url}...")
        data_dir.mkdir(parents=True, exist_ok=True)

        download_file(url, raw_file)

    logger.info(f"Loading raw data from file {raw_file}...")
    raw_df = pd.read_csv(
        raw_file, compression="gzip",
        header=None, usecols=feature_cols)

    train_ds = df_to_tensordataset(raw_df[0:train_size])
    valid_ds = df_to_tensordataset(raw_df[train_size:train_size+valid_size])
    test_ds = df_to_tensordataset(
        raw_df[train_size+valid_size:train_size+valid_size+test_size])

    return train_ds, valid_ds, test_ds
