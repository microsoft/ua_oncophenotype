import collections
from typing import Dict, List, Optional, Union, cast

from llm_lib.data_loaders.data_loaders_base import DatasetBase
from llm_lib.dtypes import ClinicalNote


class NoteRetrievalBase:
    def get_note(self, note_id: str) -> ClinicalNote:
        raise NotImplementedError()

    def get_patient_notes(
        self,
        patient_id: str,
        note_types: Optional[Union[str, List[str]]] = None,
        date_min: Optional[str] = None,
        date_max: Optional[str] = None,
    ) -> List[ClinicalNote]:
        raise NotImplementedError()


class NoteSemanticRetrievalBase(NoteRetrievalBase):
    def get_patient_notes_by_similarity(
        self,
        patient_id: str,
        query: str,
        note_types: Optional[Union[str, List[str]]] = None,
        date_min: Optional[str] = None,
        date_max: Optional[str] = None,
    ) -> List[ClinicalNote]:
        raise NotImplementedError()


class NoteRetrievalSimple(NoteRetrievalBase):
    """Simple in-memory note retrieval"""

    def __init__(self, dataset: DatasetBase):
        self.notes_by_id: Dict[str, ClinicalNote] = {
            note.id: cast(ClinicalNote, note) for note in dataset.get_items()
        }
        notes_by_patient_id = collections.defaultdict(list)
        for note in self.notes_by_id.values():
            notes_by_patient_id[note.patient_id].append(note)
        self.notes_by_patient_id: Dict[str, List[ClinicalNote]] = dict(
            notes_by_patient_id
        )

    def get_note(self, note_id: str) -> ClinicalNote:
        return self.notes_by_id[note_id]

    def get_patient_notes(
        self,
        patient_id: str,
        note_types: Optional[Union[str, List[str]]] = None,
        date_min: Optional[str] = None,
        date_max: Optional[str] = None,
    ) -> List[ClinicalNote]:
        notes = self.notes_by_patient_id[patient_id]
        if note_types is not None:
            if isinstance(note_types, str):
                note_types = [note_types]
            notes = [note for note in notes if note.type in note_types]
        if date_min is not None:
            notes = [note for note in notes if note.date >= date_min]
        if date_max is not None:
            notes = [note for note in notes if note.date <= date_max]
        return notes
