"""Test serialization / deserialization of db_types"""
import pytest

from ..dtypes import Annotation, ClinicalNote, Document, Patient


def test_serialization():
    # basic serialization / deserialization test
    note = ClinicalNote(
        id="foo_id",
        type="Document.ClinicalNote",
        date="2022-01-01T00:00:00Z",
        patient_id="some_pat_id",
        text="some text",
    )
    d = note.to_dict()
    print(d)
    new_note = ClinicalNote.from_dict(d)
    print(vars(note))
    print(vars(new_note))
    assert new_note.__class__ == ClinicalNote
    assert new_note == note


def test_metadata_deserialization():
    # show that extra fields are put in metadata
    note = ClinicalNote(
        id="foo_id",
        type="Document.ClinicalNote",
        date="2022-01-01T00:00:00Z",
        patient_id="some_pat_id",
        text="some text",
    )
    d = note.to_dict()
    d["foo"] = "bar"
    print(d)
    new_note = ClinicalNote.from_dict(d)
    assert new_note.__class__ == ClinicalNote
    assert new_note.metadata["foo"] == "bar"


if __name__ == "__main__":
    test_serialization()
    test_metadata_deserialization()
