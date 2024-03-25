import collections
import json
from pathlib import Path
from typing import Dict, Optional

from rwd_llm.chains.patient_history_grounded_answer_chain import (
    PatientHistoryGroundedAnswer,
)
from rwd_llm.memory import PickleDeserializer
from rwd_llm.memory.persistent_kv_memory import SerializedMemoryValue

DESERIALIZER = PickleDeserializer()


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
    patient_history = "Summarized Patient History:\n\n"
    evidence_by_note_id = collections.defaultdict(list)
    for evidence in p.evidence:
        evidence_by_note_id[evidence.note_id].append(evidence.evidence)
    for evidence in p.contradictory_evidence:
        evidence_by_note_id[evidence.note_id].append(evidence.evidence)
    for note_idx, (note_id, evidence) in enumerate(evidence_by_note_id.items()):
        patient_history += f"Note ID: {note_idx}\n----------\n"
        for e in evidence:
            patient_history += f"  - {e}\n"
    encoded_result = p.json()

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
    memory = SerializedMemoryValue.parse_obj(obj)
    value: PatientHistoryGroundedAnswer = DESERIALIZER(memory.value)
    return grounded_answer_to_example(
        value,
        example_patient_history_key,
        example_result_key,
        id_key=id_key,
        patient_id=patient_id,
    )
