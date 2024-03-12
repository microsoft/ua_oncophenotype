from ...memory.persistent_kv_memory import FileMemoryProvider
from ...memory.pickle_serializer import PickleSerializer
from ..memory_example_selector import MemoryExampleSelector
from rwd_llm.chains.patient_history_grounded_answer_chain import (
    PatientHistoryGroundedAnswer,
    PatientHistoryEvidence,
)
from tempfile import TemporaryDirectory

import pandas as pd


def test_memory_example_selector():
    RUN_ID = "run-1"
    ser = PickleSerializer()
    # create a temporary directory for storing test memories
    with TemporaryDirectory() as temp_dir:

        # set up some memories to use as examples
        mp = FileMemoryProvider.initialize(run_id=RUN_ID, persistence_dir=temp_dir)
        # we expect serialized objects when we read from memory
        PatientHistoryGroundedAnswer(
            evidence=[
                PatientHistoryEvidence(note_id="1", evidence="evidence_a_1"),
                PatientHistoryEvidence(note_id="2", evidence="evidence_a_2"),
            ],
            reasoning="because...",
            contradictory_evidence=[],
            answer="answer_a",
        )
        mp.add_memory("item1", "mem1", ser("val_a"))
        mp.add_memory("item1", "mem2", ser("val_b"))
        mp.add_memory("item2", "mem1", ser("val_c"))

        # create a memory example selector
        memory_example_selector = MemoryExampleSelector(
            memory_dir=temp_dir,
            memory_name="mem1",
            n_examples=3,
        )
        # select examples
        examples = memory_example_selector.select_examples({})
        # assert that we get all the examples
        assert set(examples) == {"val_a", "val_c"}

        meta_df = pd.DataFrame(
            {
                "id": ["item1", "item2"],
                "filter_col": ["A", "B"],
            }
        )

        # create a memory example selector
        memory_example_selector = MemoryExampleSelector(
            memory_dir=temp_dir,
            memory_name="mem1",
            n_examples=3,
            example_metadata_path=meta_df,
            id_column="id",
            filter_cols=["filter_col"],
        )

        # select examples
        examples = memory_example_selector.select_examples({"filter_col": "A"})
        # assert that the examples are empty
        assert set(examples) == {"val_a"}


if __name__ == "__main__":
    test_memory_example_selector()
