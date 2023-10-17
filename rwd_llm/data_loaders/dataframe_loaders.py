import os
from typing import Dict, List, Optional

import pandas as pd
from llm_lib.dtypes.dtypes import ClinicalNote, Label, Patient

from .data_loaders_base import DataLoaderBase, DatasetBase


def _load_dataframe(
    dataframe_path,
    data_root_dir: Optional[str],
) -> pd.DataFrame:
    if data_root_dir:
        # allow override of data location
        dataframe_path = os.path.join(data_root_dir, dataframe_path)
    _, file_ext = os.path.splitext(dataframe_path)
    if file_ext.lower() == ".json":
        df = pd.read_json(dataframe_path)
    elif file_ext.lower() == ".csv":
        df = pd.read_csv(dataframe_path)
    elif file_ext.lower() == ".tsv":
        df = pd.read_csv(dataframe_path, sep="\t")
    else:
        raise ValueError(f"Unrecognized file extension: {file_ext}")

    df = df.astype(str)

    return df


def _get_split_df(
    df: pd.DataFrame, split_column: Optional[str], split_name: Optional[str]
) -> pd.DataFrame:
    """Return sub-dataframe for the specified split name.  If split_name is None, just
    return the original dataframe."""

    if split_column and split_column not in df.columns:
        raise ValueError(f"Split column {split_column} not found in dataframe")

    if split_name:
        if not split_column:
            raise ValueError(
                "Cannot specify split name if split column is not provided"
            )
        split_names = set(df[split_column])
        if split_name not in split_names:
            raise ValueError(
                f"Split name {split_name} not found in column {split_column}"
            )

        # select only instances with the specified split name
        df = df[df[split_column] == split_name]

    return df


def _load_notes(
    df: pd.DataFrame,
    text_column: str,
    date_column: str,
    patient_id_column: str,
    type_column: Optional[str],
    note_id_column: Optional[str],
    note_type: str,
) -> List[ClinicalNote]:
    columns = set(df.columns)

    if text_column not in columns:
        raise ValueError(f"Text column {text_column} not found in dataframe")

    if date_column not in columns:
        raise ValueError(f"Date column {date_column} not found in dataframe")

    if type_column and type_column not in columns:
        raise ValueError(f"Type column {type_column} not found in dataframe")

    if note_id_column and note_id_column not in columns:
        raise ValueError(f"ID column {note_id_column} not found in dataframe")

    notes = []
    for ridx, row in df.iterrows():
        note = ClinicalNote(
            text=row[text_column],
            date=row[date_column],
            patient_id=row[patient_id_column],
            id=row[note_id_column] if note_id_column else str(ridx),
            type=row[type_column] if type_column else note_type,
        )
        notes.append(note)
    return notes


def _load_labels(
    df: pd.DataFrame,
    id_column: Optional[str],
    label_column: str,
    label_type: str = "Label",
) -> Dict[str, Label]:
    labels: Dict[str, Label] = {}
    for ridx, row in df.iterrows():
        _id = row[id_column] if id_column else str(ridx)
        cur_label = labels.get(_id, None)
        new_label = row[label_column]
        if cur_label is not None and cur_label.label != new_label:
            raise ValueError(
                f"Found conflicting labels for item {_id}: {cur_label} and {new_label}"
            )
        labels[_id] = Label(
            id=_id,
            type=label_type,
            label=new_label,
        )
    return labels


class DataframeClinicalNoteDataLoader(DataLoaderBase):
    def __init__(
        self,
        dataframe_path: str,
        text_column: str,
        date_column: str,
        patient_id_column: str,
        split_column: Optional[str] = None,
        type_column: Optional[str] = None,
        note_id_column: Optional[str] = None,
        note_type: str = "Document.ClinicalNote",
        data_root_dir: Optional[str] = None,
        label_column: Optional[str] = None,
    ):
        self.dataframe_path = dataframe_path
        self.text_column = text_column
        self.date_column = date_column
        self.patient_id_column = patient_id_column
        self.split_column = split_column
        self.note_id_column = note_id_column
        self.type_column = type_column
        self.note_type = note_type
        self.data_root_dir = data_root_dir
        self.label_column = label_column

    def load(self, split_name: Optional[str] = None) -> DatasetBase:
        df = _load_dataframe(self.dataframe_path, self.data_root_dir)
        df = _get_split_df(df, self.split_column, split_name)

        notes = _load_notes(
            df,
            text_column=self.text_column,
            date_column=self.date_column,
            patient_id_column=self.patient_id_column,
            type_column=self.type_column,
            note_id_column=self.note_id_column,
            note_type=self.note_type,
        )
        labels = None
        if self.label_column:
            labels = _load_labels(
                df,
                id_column=self.note_id_column,
                label_column=self.label_column,
            )
        return DatasetBase(
            items=notes,
            labels=labels,
        )


class DataframePatientDataLoader(DataLoaderBase):
    def __init__(
        self,
        dataframe_path: str,
        patient_id_column: str,
        split_column: Optional[str] = None,
        data_root_dir: Optional[str] = None,
        label_column: Optional[str] = None,
    ):
        self.dataframe_path = dataframe_path
        self.patient_id_column = patient_id_column
        self.split_column = split_column
        self.data_root_dir = data_root_dir
        self.label_column = label_column

    def load(self, split_name: Optional[str] = None) -> DatasetBase:
        df = _load_dataframe(self.dataframe_path, self.data_root_dir)
        df = _get_split_df(df, self.split_column, split_name)

        patient_ids = set(df[self.patient_id_column].unique())
        patients = [Patient(id=p_id, type="Patient") for p_id in patient_ids]

        labels = None
        if self.label_column:
            labels = _load_labels(
                df,
                id_column=self.patient_id_column,
                label_column=self.label_column,
            )
        return DatasetBase(
            items=patients,
            labels=labels,
        )


def get_dataset_from_loader(
    dataset_loader: DataLoaderBase, split_name: Optional[str] = None
) -> DatasetBase:
    return dataset_loader.load(split_name=split_name)
