from rwd_llm.dtypes import ClinicalNote
from rwd_llm.tests.utils import FakeLLM

from ..patient_history_grounded_answer_chain import (
    ParsedPatientHistoryEvidence,
    PatientHistoryEvidence,
    PatientHistoryGroundedAnswer,
    PatientHistoryGroundedAnswerChain,
)
from ..retrieval_chain import note_to_document


def test_normal_usage():
    # set up the LLM to return the expected answer in JSON format
    expected_answer = PatientHistoryGroundedAnswer(
        reasoning="There isn't any useful information",
        answer="a",
        evidence=[PatientHistoryEvidence(note_id="0", evidence="First note.")],
        contradictory_evidence=[],
    )
    FakeLLM.set_answers([expected_answer.json()])
    llm = FakeLLM()

    # set up the chain
    question = """
    Based on the following patient history, what is the patient's diagnosis?
    
    Patient History:

    {pat_hist}

    Use the following format for the answer:

    {output_fmt}

    """

    chain = PatientHistoryGroundedAnswerChain.from_openai_llm(
        llm=llm,
        question=question,
        input_variables=["pat_hist"],
        patient_history_key="pat_hist",
        summarized_notes_key="sum_notes",
        preamble="This is a preamble.",
        format_instructions_variable="output_fmt",
        memory=None,
        output_key="answer_raw",
        answer_key="answer",
        parsed_answer_key="parsed_answer",
        label_mapping=["a", "b", "c"],
    )

    # set up the inputs

    summarized_notes = [
        ClinicalNote(
            id="note_001",
            type="ImagingReport",
            patient_id="pat_001",
            date="2022-01-01T00:00:00",
            text="First note.",
        ),
        ClinicalNote(
            id="note_002",
            type="ImagingReport",
            patient_id="pat_002",
            date="2022-01-02T00:00:00",
            text="Second note.",
        ),
    ]
    summarized_notes = [note_to_document(note) for note in summarized_notes]
    pat_hist = """
    Note ID: 0
    Date: 2022-01-01T00:00:00
    ----------
    First note.
    ----------

    ----------------------
    Note ID: 1
    Date: 2022-01-02T00:00:00
    ----------
    Second note.
    ----------
    ----------------------
    """

    inputs = {"sum_notes": summarized_notes, "pat_hist": pat_hist}

    # run the chain

    output = chain(inputs)
    assert output["answer"] == expected_answer.answer
    parsed_answer: PatientHistoryGroundedAnswer = output["parsed_answer"]
    assert parsed_answer.answer == expected_answer.answer
    assert len(parsed_answer.evidence) == 1
    ev = parsed_answer.evidence[0]
    assert type(ev) == ParsedPatientHistoryEvidence
    assert ev.note_id == "note_001"
    assert ev.begin == 0
    assert ev.end == len(ev.evidence)


def test_jinja_template_usage():
    # set up the LLM to return the expected answer in JSON format
    expected_answer = PatientHistoryGroundedAnswer(
        reasoning="There isn't any useful information",
        answer="a",
        evidence=[PatientHistoryEvidence(note_id="0", evidence="First note.")],
        contradictory_evidence=[],
    )
    FakeLLM.set_answers([expected_answer.json()])
    llm = FakeLLM()

    # set up the chain
    question = """
    Based on the following patient history, what is the patient's diagnosis?
    
    Patient History:

    {{pat_hist}}

    {% if my_opt == "a" %}
    foo
    {% else %}
    bar
    {% endif %}

    Use the following format for the answer:

    {{output_fmt}}

    """

    chain = PatientHistoryGroundedAnswerChain.from_openai_llm(
        llm=llm,
        question=question,
        input_variables=["pat_hist", "my_opt"],
        patient_history_key="pat_hist",
        summarized_notes_key="sum_notes",
        preamble="This is a preamble.",
        format_instructions_variable="output_fmt",
        memory=None,
        output_key="answer_raw",
        answer_key="answer",
        parsed_answer_key="parsed_answer",
        label_mapping=["a", "b", "c"],
        jinja=True,
    )

    # set up the inputs

    summarized_notes = [
        ClinicalNote(
            id="note_001",
            type="ImagingReport",
            patient_id="pat_001",
            date="2022-01-01T00:00:00",
            text="First note.",
        ),
        ClinicalNote(
            id="note_002",
            type="ImagingReport",
            patient_id="pat_002",
            date="2022-01-02T00:00:00",
            text="Second note.",
        ),
    ]
    summarized_notes = [note_to_document(note) for note in summarized_notes]
    pat_hist = """
    Note ID: 0
    Date: 2022-01-01T00:00:00
    ----------
    First note.
    ----------

    ----------------------
    Note ID: 1
    Date: 2022-01-02T00:00:00
    ----------
    Second note.
    ----------
    ----------------------
    """

    inputs = {"sum_notes": summarized_notes, "pat_hist": pat_hist, "my_opt": "a"}

    # run the chain

    output = chain(inputs)
    assert output["answer"] == expected_answer.answer
    parsed_answer: PatientHistoryGroundedAnswer = output["parsed_answer"]
    assert parsed_answer.answer == expected_answer.answer
    assert len(parsed_answer.evidence) == 1
    ev = parsed_answer.evidence[0]
    assert type(ev) == ParsedPatientHistoryEvidence
    assert ev.note_id == "note_001"
    assert ev.begin == 0
    assert ev.end == len(ev.evidence)


if __name__ == "__main__":
    test_normal_usage()
    test_jinja_template_usage()
