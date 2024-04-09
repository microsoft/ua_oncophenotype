from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

import pandas as pd
from langchain_core.embeddings import Embeddings
from rwd_llm.chains.patient_history_grounded_answer_chain import (
    PatientHistoryEvidence,
    PatientHistoryGroundedAnswer,
)

from ...memory.persistent_kv_memory import FileMemoryProvider
from ...memory.pickle_serializer import PickleSerializer
from ..index_example_knn_selector import IndexExampleKNNSelector


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing. (taken from langchain)"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def get_examples() -> Dict[str, PatientHistoryGroundedAnswer]:
    examples = {}
    ex = PatientHistoryGroundedAnswer(
        evidence=[
            PatientHistoryEvidence(note_id="1", evidence=""),
            PatientHistoryEvidence(note_id="2", evidence="evidence_a_2"),
        ],
        reasoning="because...",
        contradictory_evidence=[],
        answer="answer_a",
    )
    examples["item_000"] = ex
    ex = PatientHistoryGroundedAnswer(
        evidence=[
            PatientHistoryEvidence(note_id="1", evidence=""),
            PatientHistoryEvidence(note_id="2", evidence="evidence_a_2"),
        ],
        reasoning="because...",
        contradictory_evidence=[],
        answer="answer_a",
    )
    examples["item_001"] = ex
    return examples


def _get_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["item_000", "item_001"],
            "filter_col": ["A", "B"],
        }
    )


RUN_ID = "run-1"


def _create_memories(
    mem_dir: Path, memory_name: str, examples: Dict[str, PatientHistoryGroundedAnswer]
) -> pd.DataFrame:
    ser = PickleSerializer()

    # set up some memories to use as examples
    mp = FileMemoryProvider.initialize(run_id=RUN_ID, persistence_dir=str(mem_dir))
    for example_id, example in examples.items():
        mp.add_memory(example_id, memory_name, ser(example))

    metadata = _get_metadata()
    return metadata


def _create_index(
    mem_dir: Path, index_dir: Path, memory_name: str, metadata: pd.DataFrame
) -> IndexExampleKNNSelector:
    # this index_selector will just build the index
    index_selector = IndexExampleKNNSelector(
        memory_dir=mem_dir,
        memory_name=memory_name,
        embedding_model_name=FakeEmbeddings(),
        query_text_input="evidence",
        persist_path=str(index_dir),
        example_metadata_path=metadata,
        metadata_filter_col_map={"the_metadata_col": "the_input_key"},
        id_column="id",
        filter_cols=["filter_col"],
        example_patient_history_key="ph_key",
        example_result_key="result_key",
        build_index=True,
        k=3,
    )
    # not really doing anything here, we just instantiated the index_selector to create
    # the index.
    assert isinstance(index_selector, IndexExampleKNNSelector)
    return index_selector


def test_index_example_knn_selector():
    mem_name = "mem1"
    examples = get_examples()

    with TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        mem_dir = base_dir / "memories"
        mem_dir.mkdir()
        index_dir = base_dir / "index"
        index_dir.mkdir()

        # create the memories
        metadata = _create_memories(mem_dir, memory_name=mem_name, examples=examples)

        # build the index
        _create_index(mem_dir, index_dir, memory_name=mem_name, metadata=metadata)

        # mapping from metadata column to input variable name
        meatadata_filter_col_map = {"filter_col": "input_filter_key"}

        # create a memory example selector
        index_selector = IndexExampleKNNSelector(
            memory_dir=mem_dir,
            memory_name=mem_name,
            embedding_model_name=FakeEmbeddings(),
            query_text_input="evidence",
            persist_path=str(index_dir),
            example_metadata_path=metadata,
            metadata_filter_col_map=meatadata_filter_col_map,
            id_column="id",
            filter_cols=["filter_col"],
            example_patient_history_key="ph_key",
            example_result_key="result_key",
            build_index=True,
            k=3,
        )
        # select examples
        input = {
            "evidence": "This is the string used for similarity comparison.",
            "input_filter_key": "A",
        }
        examples = index_selector.select_examples(input)

        # assert that we get the example with "input_filter_key" == "A"
        assert len(examples) == 1
        assert examples[0]["id"] == "item_000"


if __name__ == "__main__":
    test_index_example_knn_selector()
