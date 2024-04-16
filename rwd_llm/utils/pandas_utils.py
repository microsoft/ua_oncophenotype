import os
from typing import Optional

import pandas as pd


def load_dataframe(
    dataframe_path,
    data_root_dir: Optional[str] = None,
) -> pd.DataFrame:
    if data_root_dir:
        # allow override of data location
        dataframe_path = os.path.join(data_root_dir, dataframe_path)
    _, file_ext = os.path.splitext(dataframe_path)
    if file_ext.lower() == ".json":
        df = pd.read_json(dataframe_path, dtype=False)
    elif file_ext.lower() == ".csv":
        df = pd.read_csv(dataframe_path)
    elif file_ext.lower() == ".tsv":
        df = pd.read_csv(dataframe_path, sep="\t")
    elif file_ext.lower() == ".jsonl" or file_ext.lower() == "":
        df = pd.read_json(dataframe_path, dtype=False, lines=True)
    else:
        raise ValueError(f"Unrecognized file extension: {file_ext}")

    df = df.astype(str)

    return df
