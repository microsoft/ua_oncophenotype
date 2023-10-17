from typing import List

from attr import dataclass
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


@dataclass
class EvidenceChainExample:
    text: str
    answer: str
    evidence: str


_EVIDENCE_TEMPLATE_FMT = '''
Question: {question}

Source Document:
----------
{{text}}
----------
Answer [{answer_options_short}]:
{{answer}}
Evidence:
"""{{evidence}}"""
'''


def _truncate_to(full_str, sub_str):
    return full_str[: full_str.find(sub_str)]


def _truncate_from(full_str, sub_str):
    return full_str[full_str.find(sub_str) :]


def _get_evidence_template_fmt(
    question: str, answer_options: List[str] = ["Yes", "No"]
) -> str:
    answer_options_short = "/".join(answer_options)
    return _EVIDENCE_TEMPLATE_FMT.format(
        question=question, answer_options_short=answer_options_short
    )


def _get_answer_prefix_fmt(
    question: str, answer_options: List[str] = ["Yes", "No"]
) -> str:
    evidence_template_fmt = _get_evidence_template_fmt(question, answer_options)
    return _truncate_to(evidence_template_fmt, "{answer}")


ANSWER_STOP = ["\n", "<|im_end|>"]


def _get_evidence_prefix_fmt(
    question: str, answer_options: List[str] = ["Yes", "No"]
) -> str:
    evidence_template_fmt = _get_evidence_template_fmt(question, answer_options)
    return _truncate_to(evidence_template_fmt, "{evidence}")


EVIDENCE_STOP = ['"""', "<|im_end|>"]


# used as format for few-shot prompts
def _get_evidence_example_prompt(
    question: str, answer_options: List[str] = ["Yes", "No"]
) -> PromptTemplate:
    evidence_template_fmt = _get_evidence_template_fmt(question, answer_options)
    return PromptTemplate(
        template=evidence_template_fmt,
        input_variables=[
            "text",
            "answer",
            "evidence",
        ],
    )


DEFAULT_EXAMPLE_SEPARATOR = "\n ============= \n"


def get_answer_prompt(
    preamble: str,
    question: str,
    examples: List[EvidenceChainExample],
    answer_options: List[str] = ["Yes", "No"],
) -> FewShotPromptTemplate:
    evidence_example_prompt = _get_evidence_example_prompt(question, answer_options)
    answer_prefix_fmt = _get_answer_prefix_fmt(question, answer_options)
    return FewShotPromptTemplate(
        example_prompt=evidence_example_prompt,
        examples=[vars(example) for example in examples],
        prefix=preamble,
        suffix=answer_prefix_fmt,
        example_separator=DEFAULT_EXAMPLE_SEPARATOR,
        input_variables=["text"],
    )


def get_evidence_prompt(
    preamble: str,
    question: str,
    examples: List[EvidenceChainExample],
    answer_options: List[str] = ["Yes", "No"],
) -> FewShotPromptTemplate:
    evidence_example_prompt = _get_evidence_example_prompt(question, answer_options)
    evidence_prefix_fmt = _get_evidence_prefix_fmt(question, answer_options)
    return FewShotPromptTemplate(
        example_prompt=evidence_example_prompt,
        examples=[vars(example) for example in examples],
        prefix=preamble,
        suffix=evidence_prefix_fmt,
        example_separator=DEFAULT_EXAMPLE_SEPARATOR,
        input_variables=["text", "answer"],
    )


#
# CoT prompt templates
#

# evidence first, then answer -- use evidence as CoT reasoning
_EVIDENCE_COT_TEMPLATE_FMT = '''
Question: {question}

Source Document:
----------
{{text}}
----------
Evidence:
"""{{evidence}}"""

Answer [{answer_options_short}]:
{{answer}}
'''


def _get_evidence_cot_template_fmt(
    question: str, answer_options: List[str] = ["Yes", "No"]
) -> str:
    answer_options_short = "/".join(answer_options)
    return _EVIDENCE_COT_TEMPLATE_FMT.format(
        question=question, answer_options_short=answer_options_short
    )


# used as format for few-shot prompts
def _get_evidence_cot_example_prompt(
    question: str, answer_options: List[str] = ["Yes", "No"]
) -> PromptTemplate:
    evidence_cot_template_fmt = _get_evidence_cot_template_fmt(question, answer_options)
    return PromptTemplate(
        template=evidence_cot_template_fmt,
        input_variables=[
            "text",
            "answer",
            "evidence",
        ],
    )


def _get_evidence_cot_prefix_fmt(
    question: str, answer_options: List[str] = ["Yes", "No"]
) -> str:
    evidence_cot_template_fmt = _get_evidence_cot_template_fmt(question, answer_options)
    return _truncate_to(evidence_cot_template_fmt, "{evidence}")


EVIDENCE_COT_STOP = [DEFAULT_EXAMPLE_SEPARATOR, "<|im_end|>"]


def get_evidence_cot_prompt(
    preamble: str,
    question: str,
    examples: List[EvidenceChainExample],
    answer_options: List[str] = ["Yes", "No"],
) -> FewShotPromptTemplate:
    evidence_cot_example_prompt = _get_evidence_cot_example_prompt(
        question, answer_options
    )
    evidence_cot_prefix_fmt = _get_evidence_cot_prefix_fmt(question, answer_options)
    return FewShotPromptTemplate(
        example_prompt=evidence_cot_example_prompt,
        examples=[vars(example) for example in examples],
        prefix=preamble,
        suffix=evidence_cot_prefix_fmt,
        example_separator=DEFAULT_EXAMPLE_SEPARATOR,
        input_variables=["text"],
    )


def get_evidence_cot_response_pattern() -> str:
    response_fmt = _truncate_from(_EVIDENCE_COT_TEMPLATE_FMT, "{{evidence}}")
    response_fmt = response_fmt.format(answer_options_short="(.+)")
    # escape literals that are regex special chars, and remove trailing newline
    _pat = response_fmt.strip().replace("[", r"\[").replace("]", r"\]")
    # then insert patterns to parse out the model outputs
    # EVIDENCE_COT_RESPONSE_PATTERN = r'(.+)"""\n+Answer \[(.+)\]:\n([^\n]+).*?'
    return _pat.format(evidence="(.+)", answer="([^\n]+).*?")
