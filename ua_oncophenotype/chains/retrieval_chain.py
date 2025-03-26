from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.documents import Document
from ua_oncophenotype.dtypes import ClinicalNote, NoteFormatterBase

from ..retrieval import NoteRetrievalBase


def note_to_document(note: ClinicalNote) -> Document:
    """Convert our clinical note type to a langchain Document."""
    note_dict = note.to_dict()
    text = note_dict.pop("text")
    metadata = note_dict.pop("metadata", {})
    metadata.update(note_dict)
    return Document(page_content=text, metadata=metadata)


class RetrievalChain(Chain):
    retriever: NoteRetrievalBase
    note_types: Optional[List[str]] = None
    convert_to_documents: bool = False
    patient_id_key: str = "patient_id"
    output_key: str = "notes"
    ascending_date_sort: bool = True

    @property
    def input_keys(self) -> List[str]:
        """Defines the input keys."""
        return [self.patient_id_key]

    @property
    def output_keys(self) -> List[str]:
        """Defines the output keys."""
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        notes = self.retriever.get_patient_notes(
            patient_id=inputs[self.patient_id_key], note_types=self.note_types
        )
        # sort by date
        notes = list(
            sorted(
                notes, key=lambda note: note.date, reverse=not self.ascending_date_sort
            )
        )

        if self.convert_to_documents:
            notes = [note_to_document(note) for note in notes]
        return {self.output_key: notes}


class RetrievalAndClassificationChain(Chain):
    retriever: NoteRetrievalBase
    classifier: Chain
    note_formatter: NoteFormatterBase
    note_types: Optional[List[str]] = None

    @property
    def input_keys(self) -> List[str]:
        """Defines the input keys."""
        return ["id"]

    @property
    def output_keys(self) -> List[str]:
        """Defines the output keys."""
        return self.classifier.output_keys + ["patient_id"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        patient_id = inputs["id"]
        notes = self.retriever.get_patient_notes(
            patient_id=patient_id, note_types=self.note_types
        )
        notes_text = self.note_formatter.format(notes)
        inputs = {"text": notes_text}

        callbacks = _run_manager.get_child()
        outputs = self.classifier(inputs, callbacks=callbacks, return_only_outputs=True)
        outputs["patient_id"] = patient_id
        return outputs
