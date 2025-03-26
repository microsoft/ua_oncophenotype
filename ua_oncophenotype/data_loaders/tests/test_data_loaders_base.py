from collections import defaultdict

from ua_oncophenotype.dtypes import ClinicalNote, Label, Patient

from ..data_loaders_base import DatasetBase


def test_patient_dataset():
    patients = [
        Patient(id="1"),
        Patient(id="2"),
        Patient(id="3"),
        Patient(id="4"),
        Patient(id="5"),
    ]
    notes = [
        ClinicalNote(
            id="p1n1", patient_id="1", type="note", text="some text", date="2020-01-01"
        ),
        ClinicalNote(
            id="p1n2", patient_id="1", type="note", text="some text", date="2020-01-01"
        ),
        ClinicalNote(
            id="p2n1", patient_id="2", type="note", text="some text", date="2020-01-01"
        ),
        ClinicalNote(
            id="p3n1", patient_id="3", type="note", text="some text", date="2020-01-01"
        ),
        ClinicalNote(
            id="p4n1", patient_id="4", type="note", text="some text", date="2020-01-01"
        ),
        ClinicalNote(
            id="p4n2", patient_id="4", type="note", text="some text", date="2020-01-01"
        ),
        ClinicalNote(
            id="p5n1", patient_id="5", type="note", text="some text", date="2020-01-01"
        ),
    ]
    labels = [
        Label(id="1", type="label", label="No"),
        Label(id="2", type="label", label="No"),
        Label(id="3", type="label", label="Yes"),
        Label(id="4", type="label", label="Yes"),
        Label(id="5", type="label", label="No"),
    ]
    labels_by_id = {label.id: label for label in labels}
    notes_by_patient_id = defaultdict(list)
    for note in notes:
        notes_by_patient_id[note.patient_id].append(note)
    dataset = DatasetBase(
        items=patients,
        labels=labels_by_id,
    )
    assert len(dataset) == 5
    assert len(dataset[0:2]) == 2

    small_dataset = dataset.subset_by_ids(["1", "3"])
    assert len(small_dataset) == 2
    assert len(small_dataset.labels) == 2

    assert len(dataset) == 5
    assert len(dataset.labels) == 5

    dataset.subset_by_ids_inplace(["1", "3"])
    assert len(dataset) == 2
    assert len(dataset.labels) == 2


if __name__ == "__main__":
    test_patient_dataset()
