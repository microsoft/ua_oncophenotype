import textwrap

from langchain.schema import Document
from rwd_llm.dtypes import ClinicalNote
from rwd_llm.tests.utils import FakeLLM

from ..retrieval_chain import note_to_document
from ..structured_summary import (
    ParsedSummaryEvidence,
    StructuredDocumentSummaryChain,
    Summary,
    SummaryEvidence,
    SummaryFinding,
)


class FakeOpenAILLM(FakeLLM):
    model_name: str


def test_zero_shot_usage():
    input_text = textwrap.dedent("""
    This is a note to summarize.  It has a few findings.  One is that the Patient has a
    broken leg.  Another is that the Patient has a broken arm.  Also, the Patient is
    allergic to peanuts.
    """)
    findings = [
        {
            "finding": "broken leg",
            "evidence": "the patient has a broken leg",
            "exact_span": "the Patient has a\nbroken leg",
        },
        {
            "finding": "broken arm",
            "evidence": "the patient has a broken arm",
            "exact_span": "the Patient has a broken arm",
        },
        {
            "finding": "peanut allergy",
            "evidence": "the patient is allergic to peanuts",
            "exact_span": "the Patient is\nallergic to peanuts",
        },
    ]
    # set up the LLM to return the expected answer in JSON format
    expected_answer = Summary(
        title="This is a title.",
        findings=[
            SummaryFinding(
                finding=finding["finding"],
                evidence=[SummaryEvidence(evidence=finding["evidence"])],
            )
            for finding in findings
        ],
    )
    FakeLLM.set_answers([expected_answer.json()])
    llm = FakeOpenAILLM(model_name="gpt-4")

    chain = StructuredDocumentSummaryChain.from_openai_llm(
        llm=llm,
        question="Summarize the document: {text}.  Use format {format_instructions}",
        input_variables=["text"],
        format_instructions_variable="format_instructions",
        doc_key="doc",
        prompt_doc_key="text",
        output_key="summary",
        truncate=False,
    )

    # set up the chain
    clinical_note = ClinicalNote(
        id="note_001",
        type="ImagingReport",
        text=input_text,
        date="2022-01-01T00:00:00",
        patient_id="pat_001",
    )
    input_doc = note_to_document(clinical_note)

    # set up the inputs
    inputs = {"doc": input_doc}

    # run the chain

    output = chain(inputs)
    out_doc: Document = output["summary"]
    summary: Summary = out_doc.metadata["summary"]
    for fidx, finding in enumerate(summary.findings):
        for evidence in finding.evidence:
            assert type(evidence) == ParsedSummaryEvidence
            assert evidence.begin is not None
            assert evidence.end is not None
            assert (
                input_text[evidence.begin : evidence.end]
                == findings[fidx]["exact_span"]
            )
            print(input_text[evidence.begin : evidence.end])


def test_few_shot_usage():
    input_text = textwrap.dedent("""
    This is a note to summarize.  It has a few findings.  One is that the Patient has a
    broken leg.  Another is that the Patient has a broken arm.  Also, the Patient is
    allergic to peanuts.
    """)
    findings = [
        {
            "finding": "broken leg",
            "evidence": "the patient has a broken leg",
            "exact_span": "the Patient has a\nbroken leg",
        },
        {
            "finding": "broken arm",
            "evidence": "the patient has a broken arm",
            "exact_span": "the Patient has a broken arm",
        },
        {
            "finding": "peanut allergy",
            "evidence": "the patient is allergic to peanuts",
            "exact_span": "the Patient is\nallergic to peanuts",
        },
    ]
    # set up the LLM to return the expected answer in JSON format
    expected_answer = Summary(
        title="This is a title.",
        findings=[
            SummaryFinding(
                finding=finding["finding"],
                evidence=[SummaryEvidence(evidence=finding["evidence"])],
            )
            for finding in findings
        ],
    )
    examples = [
        {
            "text": "A simple note with one finding, that the patient has asthma.",
            "result": {
                "title": "simple note",
                "findings": [
                    {
                        "finding": "asthma",
                        "evidence": [{"evidence": "the patient has athsma."}],
                    }
                ],
            },
        },
        {
            "text": "A bone scan indicating osteoporosis.",
            "result": {
                "title": "bone scan",
                "findings": [
                    {
                        "finding": "osteoperosis",
                        "evidence": [
                            {"evidence": "A bone scan indicating osteoporosis."}
                        ],
                    }
                ],
            },
        },
    ]
    Summary.parse_obj(examples[1]["result"])
    FakeLLM.set_answers([expected_answer.json()])
    llm = FakeOpenAILLM(model_name="gpt-4")

    chain = StructuredDocumentSummaryChain.from_openai_llm(
        llm=llm,
        preamble="You are an AI trained to summarize clinical documents.",
        instructions=(
            "Summarize the given document as a title and list of findings. Use the"
            " output format {format_instructions}."
        ),
        question="Summarize the document:\n\n{text}",
        input_variables=["text"],
        format_instructions_variable="format_instructions",
        doc_key="doc",
        prompt_doc_key="text",
        output_key="summary",
        truncate=False,
        few_shot_examples=examples,
        example_output_var="result",
    )

    # set up the chain
    clinical_note = ClinicalNote(
        id="note_001",
        type="ImagingReport",
        text=input_text,
        date="2022-01-01T00:00:00",
        patient_id="pat_001",
    )
    input_doc = note_to_document(clinical_note)

    # set up the inputs
    inputs = {"doc": input_doc}

    # run the chain

    output = chain(inputs)
    out_doc: Document = output["summary"]
    summary: Summary = out_doc.metadata["summary"]
    for fidx, finding in enumerate(summary.findings):
        for evidence in finding.evidence:
            assert type(evidence) == ParsedSummaryEvidence
            assert evidence.begin is not None
            assert evidence.end is not None
            assert (
                input_text[evidence.begin : evidence.end]
                == findings[fidx]["exact_span"]
            )
            print(input_text[evidence.begin : evidence.end])


if __name__ == "__main__":
    test_zero_shot_usage()
    test_few_shot_usage()
