import pandas as pd
import pytest
from ua_oncophenotype.data_loaders.data_loaders_base import DatasetBase
from ua_oncophenotype.dtypes.dtypes import ClinicalNote
from ua_oncophenotype.llms import setup_openai_from_dotenv

from ..build_index import BuildIndex

setup_openai_from_dotenv(raise_error=False)


def build_df() -> pd.DataFrame:
    """
    Builds and returns a sample dataframe.
    returns: sample dataframe
    """
    return pd.DataFrame(
        {
            "text": [
                "This is a test text.",
                "This is another test text.",
                "This is a third test text.",
            ],
            "id": [1, 2, 3],
            "patient_id": [1, 1, 2],
            "sex": ["M", "M", "F"],
        }
    )


def build_dataset(df: pd.DataFrame) -> DatasetBase:
    """
    Builds a dataset from the given dataframe.
    @df: dataframe to build the dataset from
    returns: DatasetBase
    """

    return DatasetBase(
        notes=[
            ClinicalNote(
                id=row["id"],
                type="Document.ClinicalNote",
                text=row["text"],
                date="2020-01-01",
                patient_id=row["patient_id"],
                metadata={"sex": row["sex"]},
            )
            for _, row in df.iterrows()
        ]
    )


@pytest.mark.skip("requires openai calls")
def test_build_index() -> None:
    """
    Tests building the index, should output a tmp folder with the index.
    """
    df = build_df()
    dataset = build_dataset(df)
    index_builder = BuildIndex(
        dataset=dataset,
        embedding_model_name="text-embedding-ada-002",
        chunk_size=10,
        chunk_overlap=5,
        index_output_dir="/home/azureuser/tmp",
    )
    chroma = index_builder.get_index()
    assert chroma is not None
    print("length of chromadb", chroma._collection.count())


if __name__ == "__main__":
    test_build_index()
