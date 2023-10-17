import os

import pytest
from rwd_llm.llms import setup_openai_from_dotenv

from ..index_wrapper import IndexWrapper

setup_openai_from_dotenv(raise_error=False)


@pytest.mark.skip("requires openai calls")
def test_query_index() -> None:
    """
    Tests querying the index
    """
    if not os.path.exists("./tmp"):  # build index if it doesn't exist
        from .test_build_index import test_build_index

        test_build_index()

    loader = IndexWrapper("./tmp", "text-embedding-ada-002")
    docs = loader._query(
        query="This is a test query.",
        search_mode="similarity",
        k=1,
        fetch_k=20,
        filter={"id": "2"},
        filter_document={"$contains": "another"},
    )

    assert docs is not None
    print(docs)


if __name__ == "__main__":
    test_query_index()
