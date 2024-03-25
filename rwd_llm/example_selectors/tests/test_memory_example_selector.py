from tempfile import TemporaryDirectory

import pandas as pd
from rwd_llm.chains.patient_history_grounded_answer_chain import (
    PatientHistoryEvidence,
    PatientHistoryGroundedAnswer,
)

from ...memory.persistent_kv_memory import FileMemoryProvider
from ...memory.pickle_serializer import PickleSerializer
from ..memory_example_selector import MemoryExampleSelector


def test_memory_example_selector():
    RUN_ID = "run-1"
    example_result_key = "the_result"
    ID_COL = "id"

    ser = PickleSerializer()
    # create a temporary directory for storing test memories
    with TemporaryDirectory() as temp_dir:
        # set up some memories to use as examples
        mp = FileMemoryProvider.initialize(run_id=RUN_ID, persistence_dir=temp_dir)
        # we expect serialized objects when we read from memory
        answer = PatientHistoryGroundedAnswer(
            evidence=[
                PatientHistoryEvidence(note_id="1", evidence="evidence_a_1"),
                PatientHistoryEvidence(note_id="2", evidence="evidence_a_2"),
            ],
            reasoning="because...",
            contradictory_evidence=[],
            answer="answer_a",
        )
        mp.add_memory("item1", "mem1", ser(answer))
        answer = PatientHistoryGroundedAnswer(
            evidence=[
                PatientHistoryEvidence(note_id="1", evidence="evidence_a_1"),
                PatientHistoryEvidence(note_id="2", evidence="evidence_a_2"),
            ],
            reasoning="because...",
            contradictory_evidence=[],
            answer="answer_b",
        )
        mp.add_memory("item1", "mem2", ser(answer))
        answer = PatientHistoryGroundedAnswer(
            evidence=[
                PatientHistoryEvidence(note_id="1", evidence="evidence_a_1"),
                PatientHistoryEvidence(note_id="2", evidence="evidence_a_2"),
            ],
            reasoning="because...",
            contradictory_evidence=[],
            answer="answer_c",
        )
        mp.add_memory("item2", "mem1", ser(answer))

        # create a memory example selector
        memory_example_selector = MemoryExampleSelector(
            memory_dir=temp_dir,
            memory_name="mem1",
            n_examples=3,
            example_result_key=example_result_key,
        )
        # select examples
        examples = memory_example_selector.select_examples({})
        # assert that we get all the examples
        answers = {ex[ID_COL] for ex in examples}
        assert answers == {"item1", "item2"}

        meta_df = pd.DataFrame(
            {
                ID_COL: ["item1", "item2"],
                "filter_col": ["A", "B"],
            }
        )

        # create a memory example selector
        memory_example_selector = MemoryExampleSelector(
            memory_dir=temp_dir,
            memory_name="mem1",
            n_examples=3,
            example_metadata_path=meta_df,
            id_column=ID_COL,
            filter_cols=["filter_col"],
            example_result_key=example_result_key,
        )

        # select examples
        examples = memory_example_selector.select_examples({"filter_col": "A"})
        # assert that the examples are empty
        answers = {ex[ID_COL] for ex in examples}
        assert answers == {"item1"}


if __name__ == "__main__":
    test_memory_example_selector()
