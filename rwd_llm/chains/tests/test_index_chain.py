import os

import pandas as pd
import pytest
from rwd_llm.data_loaders.data_loaders_base import DatasetBase
from rwd_llm.dtypes.dtypes import ClinicalNote
from rwd_llm.retrieval.retrieval_utils import build_and_get_index
from rwd_llm.tests.utils import FakeLLM

from ..index_chain import IndexChain


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
        items=[
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


@pytest.mark.skip(reason="Requires OpenAI API calls to create index")
def test_index_chain():
    # need to set OPENAI_API_KEY to something, otherwise it will fail
    os.environ["OPENAI_API_KEY"] = "fake"

    df = build_df()
    dataset = build_dataset(df)
    index = build_and_get_index(
        dataset=dataset,
        embedding_model_name="text-embedding-ada-002",
        chunk_size=10,
        chunk_overlap=5,
        index_output_dir="./tmp",
    )

    FakeLLM.set_answers(["A useless response."])
    index_chain = IndexChain.from_openai_llm(
        index=index,
        preamble="You are a helpful chatbot.",
        question="What is the answer to the question?",
        examples=[],
        llm_class=FakeLLM,
        llm_extra_args={"deployment_name": "gpt-35-turbo"},
    )
    result = index_chain({"query": "test", "search_mode": "similarity", "k": 2})
    print(result)


if __name__ == "__main__":
    test_index_chain()
