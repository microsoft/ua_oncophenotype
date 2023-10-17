from llm_lib.tests.utils import FakeLLM

from ..categorical_chain import CategoricalChain, normalize_label_mapping, parse_output
from ..chain_utils import get_prompt_template

# Sample data
sample_doc_text = (
    "This is a sample document text. It contains some important information. The answer"
    " to the question is 42."
)


def test_output_parsing():
    labels = ["N1", "N1a", "N1a(i+)"]
    label_mapping = normalize_label_mapping(labels)

    input = "N1a(i+)"
    assert parse_output(label_mapping, input) == "N1a(i+)"
    input = '"N1a(i+)"'
    assert parse_output(label_mapping, input) == "N1a(i+)"
    input = "The answer is N1a(i+)"
    assert parse_output(label_mapping, input) == "N1a(i+)"
    input = "The answer is N1."
    assert parse_output(label_mapping, input) == "N1"
    input = "ajcc: t2bn1am0"
    assert parse_output(label_mapping, input) == "N1a"
    # test that the earliest match is returned, even if shorter
    input = "The answer is N1 (N1a)"
    assert parse_output(label_mapping, input) == "N1"


def test_normal_usage():
    labels = ["N1", "N1a", "N1b"]
    llm_output = "TNM Stage T1aN1bM0"
    FakeLLM.set_answers([llm_output])

    prompt = get_prompt_template(
        "{text}\n\nWhat is the 'N' TNM stage?", input_variables=["text"]
    )
    cat_chain = CategoricalChain(
        prompt=prompt,
        llm=FakeLLM(),
        output_key="output",
        label_key="label",
        label_mapping=labels,
    )

    result = cat_chain({"text": sample_doc_text})
    assert result["label"] == "N1b"
    assert result["output"] == llm_output


if __name__ == "__main__":
    test_output_parsing()
    test_normal_usage()
