import json
from pathlib import Path
from typing import Dict, List, Optional

from ua_oncophenotype.chains.patient_history_grounded_answer_chain import (
    PatientHistoryEvidence,
    PatientHistoryGroundedAnswer,
)
from ua_oncophenotype.memory import PickleDeserializer
from ua_oncophenotype.memory.persistent_kv_memory import SerializedMemoryValue

DESERIALIZER = PickleDeserializer()


def _to_unparsed_patient_history(
    p: PatientHistoryGroundedAnswer,
    note_id_to_idx: Dict[str, int],
) -> PatientHistoryGroundedAnswer:
    """
    Create unparsed vrsion of PatientHistoryGroundedAnswer. Also turn note_id into
    note_idx.
    """
    ph = PatientHistoryGroundedAnswer(
        evidence=[
            PatientHistoryEvidence(
                note_id=str(note_id_to_idx[e.note_id]),
                evidence=e.evidence,
            )
            for e in p.evidence
        ],
        reasoning=p.reasoning,
        contradictory_evidence=[
            PatientHistoryEvidence(
                note_id=str(note_id_to_idx[e.note_id]),
                evidence=e.evidence,
            )
            for e in p.contradictory_evidence
        ],
        answer=p.answer,
    )
    return ph


def grounded_answer_to_example(
    p: PatientHistoryGroundedAnswer,
    example_patient_history_key: str,
    example_result_key: str,
    id_key: str = "id",
    patient_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Converts a PatientHistoryGroundedAnswer into a dictionary representing the input
    (patient history) and answer (the JSON representation of the
    PatientHistoryGroundedAnswer). The patient history is just a formatted version of
    the evidence cited by the PatientHistoryGroundedAnswer.
    """
    note_id_to_idx: Dict[str, int] = {}
    patient_history = "Summarized Patient History:\n\n"
    evidence_by_note: List[List[str]] = []
    for evidence in p.evidence + p.contradictory_evidence:
        note_id = evidence.note_id
        if note_id not in note_id_to_idx:
            note_id_to_idx[note_id] = len(note_id_to_idx)
            evidence_by_note.append([])
        note_idx = note_id_to_idx[note_id]
        evidence_by_note[note_idx].append(evidence.evidence)
    for note_idx, evidence in enumerate(evidence_by_note):
        patient_history += f"Note ID: {note_idx}\n----------\n"
        for e in evidence:
            patient_history += f"  - {e}\n"
    p = _to_unparsed_patient_history(p, note_id_to_idx)
    encoded_result = p.model_dump_json()

    example = {
        example_patient_history_key: patient_history,
        example_result_key: encoded_result,
    }
    if patient_id:
        example[id_key] = patient_id
    return example


def grounded_answer_memory_to_example(
    memory_file: Path,
    example_patient_history_key: str,
    example_result_key: str,
    id_key: str = "id",
    patient_id: Optional[str] = None,
) -> Dict[str, str]:
    obj = json.loads(memory_file.read_text())
    memory = SerializedMemoryValue.model_validate(obj)
    value: PatientHistoryGroundedAnswer = DESERIALIZER(memory.value)
    return grounded_answer_to_example(
        value,
        example_patient_history_key,
        example_result_key,
        id_key=id_key,
        patient_id=patient_id,
    )
