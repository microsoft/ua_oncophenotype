import textwrap

from llm_lib.tests.utils import FakeLLM

from ..evidence_chain import Evidence, EvidenceChain, find_evidence

# Sample data
sample_doc_text = (
    "This is a sample document text. It contains some important information. The answer"
    " to the question is 42."
)


def test_normal_usage():
    FakeLLM.set_answers(["42", "The answer to the question is 42."])
    # Instantiate the EvidenceChain class
    evidence_chain = EvidenceChain.from_openai_llm(
        preamble="You are a helpful chatbot.",
        question="What is the answer to the question?",
        answer_mapping=["42", "17", "100", "1"],
        examples=[],
        llm_class=FakeLLM,
        llm_extra_args={"deployment_name": "gpt-35-turbo"},
    )
    result = evidence_chain({"text": sample_doc_text})
    print(result)
    assert result["evidence"] is not None
    assert result["evidence_string"] == "The answer to the question is 42."


def test_find_evidence():
    doc_text = """
    This is a single sentence.
    This sentence
    is broken across two lines.
    This Sentence
      Has Some Strange
        Formatting.
    """
    doc_text = textwrap.dedent(doc_text)
    ev_strings = [
        ("single sentence", "single sentence"),
        (
            "this sentence is broken across two lines",
            "This sentence\nis broken across two lines",
        ),
        ("some strange formatting.", "Some Strange\n    Formatting."),
    ]

    for ev_string, ev_true_string in ev_strings:
        ev = find_evidence(ev_string, doc_text)
        assert ev is not None
        assert doc_text[ev.begin : ev.end] == ev_true_string


if __name__ == "__main__":
    test_normal_usage()
    test_find_evidence()
