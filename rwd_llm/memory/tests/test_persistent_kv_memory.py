import json
import tempfile

import pytest

from ...chains.mapping_from_traces import SummaryDeserializer
from ...chains.retrieval_chain import note_to_document
from ...chains.structured_summary import (
    Summary,
    SummaryEvidence,
    SummaryFinding,
    validate_evidence,
)
from ...dtypes import ClinicalNote
from .. import FileMemoryProvider, ForceJsonSerializer

# Mocking data to use for the tests
MOCK_MEMORY = {
    "item1": {"key1": "value1", "key2": "value2"},
    "item2": {"keyA": "valueA"},
}


def write_mock_memory_to_dir(dir_path, memory_data):
    for item_id, memories in memory_data.items():
        item_dir = dir_path / item_id
        item_dir.mkdir()
        for key, value in memories.items():
            with open(item_dir / f"{key}.json", "w") as f:
                serialized_value = {
                    "item_id": item_id,
                    "source_run": "mock_run",
                    "key": key,
                    "value": value,
                }
                json.dump(serialized_value, f)


def test_persistent_memory_provider():
    RUN_1 = "run-1"
    RUN_2 = "run-2"
    with tempfile.TemporaryDirectory() as tmpdirname:
        # test setting / getting items
        mp = FileMemoryProvider.initialize(run_id=RUN_1, persistence_dir=tmpdirname)
        mp.add_memory("item1", "key1", "i1k1")
        mp.add_memory("item1", "key2", "i1k2")
        mp.add_memory("item2", "key1", "i2k1")
        assert "i1k1" == mp.get_memory("item1", "key1")
        assert "i1k2" == mp.get_memory("item1", "key2")
        assert "i2k1" == mp.get_memory("item2", "key1")

        # test that we can initialize a new memory provider with the same persistence
        # dir
        new_mp = FileMemoryProvider.initialize(
            run_id=RUN_1, persistence_dir=tmpdirname, input_dir=tmpdirname
        )
        assert new_mp.keys(item_id="item1") == {"key1", "key2"}
        assert new_mp.keys(item_id="item2") == {"key1"}
        assert "i1k1" == new_mp.get_memory("item1", "key1")
        assert "i1k2" == new_mp.get_memory("item1", "key2")
        assert "i2k1" == new_mp.get_memory("item2", "key1")
        new_mp.add_memory("item2", "key2", "i2k2")
        assert "i2k2" == new_mp.get_memory("item2", "key2")

        # test that we can initialize a new memory provider with the same persistence
        # dir, but use a new persistence dir
        with tempfile.TemporaryDirectory() as tmpdirname2:
            mp2 = FileMemoryProvider.initialize(
                run_id=RUN_2, persistence_dir=tmpdirname2, input_dir=tmpdirname
            )
            assert "i1k1" == mp2.get_memory("item1", "key1")
            assert mp2.keys(item_id="item1") == {"key1", "key2"}
            assert mp2.keys(item_id="item2") == {"key1", "key2"}


def _get_example_summary():
    text = """
    This is a document. It has two findings. One finding is that the patient likes
    the color blue. The other finding is that the patient is diabetic.
    """
    doc = note_to_document(
        ClinicalNote(
            id="0001", patient_id="0001", type="Document", text=text, date="2000-01-01"
        )
    )
    summary = Summary(
        title="Title summarizing the purpose of the source document",
        findings=[
            SummaryFinding(
                evidence=[
                    SummaryEvidence(
                        evidence="the patient is diabetic.",
                    ),
                ],
                finding="The patient has diabetes",
            ),
            SummaryFinding(
                evidence=[
                    SummaryEvidence(
                        evidence="the patient likes the color blue.",
                    ),
                ],
                finding="The patient likes the color blue",
            ),
        ],
    )
    val_summary = validate_evidence(summary, doc)
    doc = doc.model_copy()
    summary_text = "the summary"
    doc.page_content = summary_text
    doc.metadata["summary"] = val_summary
    return doc


def test_force_to_json():
    summary = _get_example_summary()
    ser = ForceJsonSerializer()
    deser = SummaryDeserializer()
    summary_list = [summary]
    serialized = ser(summary_list)
    print(serialized)
    deserialized = deser(serialized)
    print(deserialized)


if __name__ == "__main__":
    test_persistent_memory_provider()
    test_force_to_json()
